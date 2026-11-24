package realtimify

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

func TestSentenceBufferSplits(t *testing.T) {
	var b sentenceBuffer
	got := b.push("你好，", 40)
	require.Empty(t, got, "a short clause after one comma waits for more")
	got = b.push("这是一次语音合成测试。后面还有", 40)
	require.Equal(t, []string{"你好，这是一次语音合成测试。"}, got)
	require.Equal(t, "后面还有", b.flush())
}

func TestSentenceBufferLongRun(t *testing.T) {
	var b sentenceBuffer
	got := b.push(strings.Repeat("a", 50), 20)
	require.Len(t, got, 2)
	for _, clause := range got {
		require.LessOrEqual(t, len(clause), 20)
	}
}

func TestPipelineRespondStreamsClauseByClause(t *testing.T) {
	llm := &fakeLLM{fragments: []string{"第一句话。", "第二句", "话！"}}
	var spoken []string
	tts := &fakeTTS{onSpeak: func(text string) { spoken = append(spoken, text) }}
	p := NewPipeline(Config{LLM: llm, TTS: tts})
	sink := &recordingSink{}

	require.NoError(t, p.Respond(context.Background(), "你好", sink))

	require.Equal(t, []string{"第一句话。", "第二句话！"}, spoken)
	require.Equal(t, []string{"第一句话。", "第二句话！"}, sink.assistantText)
	require.True(t, sink.turnComplete)
	require.Equal(t, 1, len(sink.usageFrames))
	require.Equal(t, []LLMMessage{{Role: "user", Text: "你好"}, {Role: "assistant", Text: "第一句话。第二句话！"}}, p.history)
}

func TestPipelineToolCallResumesWithResults(t *testing.T) {
	llm := &fakeLLM{
		fragments: []string{"稍等。"},
		toolCalls: []FunctionCall{{ID: "c1", Name: "book", Args: json.RawMessage(`{"hour":15}`)}},
	}
	tts := &fakeTTS{}
	p := NewPipeline(Config{LLM: llm, TTS: tts})
	sink := &recordingSink{}

	require.NoError(t, p.Respond(context.Background(), "订会议室", sink))
	require.Len(t, sink.toolCalls, 1)
	require.False(t, sink.turnComplete, "a tool call is not a turn boundary")

	llm.toolCalls = nil
	llm.fragments = []string{"订好了。"}
	require.NoError(t, p.SubmitToolResults(context.Background(), []FunctionResponse{{
		ID: "c1", Name: "book", Response: json.RawMessage(`{"ok":true}`),
	}}, sink))

	require.True(t, sink.turnComplete)
	require.Equal(t, []string{"稍等。", "订好了。"}, sink.assistantText)
	// The function-call turn and the function-response turn are replayed to the
	// model on the next request.
	require.Equal(t, "订会议室", llm.lastReq.History[0].Text)
	require.Len(t, llm.lastReq.History[1].ToolCalls, 1)
	require.Len(t, llm.lastReq.History[2].ToolResults, 1)
	require.Empty(t, llm.lastReq.Prompt, "tool results carry the turn; there is no new prompt")
}

func TestPipelineBargeInStopsAudio(t *testing.T) {
	llm := &fakeLLM{fragments: []string{"第一句。", "第二句。", "第三句。"}}
	tts := &fakeTTS{onSpeak: func(string) { time.Sleep(10 * time.Millisecond) }}
	p := NewPipeline(Config{LLM: llm, TTS: tts})

	ctx, cancel := context.WithCancel(context.Background())
	sink := &recordingSink{onAudio: func() { cancel() }}
	err := p.Respond(ctx, "你好", sink)
	require.ErrorIs(t, err, context.Canceled)
	require.Len(t, sink.audio, 1, "no audio may follow the interruption")
}

func TestPipelineTranscriptionDeltas(t *testing.T) {
	factory := &fakeASRFactory{}
	p := NewPipeline(Config{ASR: factory})
	sink := &recordingSink{}
	ctx := context.Background()

	require.NoError(t, p.PushAudio(ctx, []byte{1, 2}, sink))
	require.NoError(t, p.PushAudio(ctx, []byte{3, 4}, sink))
	require.True(t, factory.opened == 1, "one recognition stream per utterance")

	session := factory.session
	require.NoError(t, session.partial("你好"))
	require.NoError(t, session.partial("你好呀"))
	text, err := p.CommitUserTurn(ctx, sink)
	require.NoError(t, err)
	require.Equal(t, "你好呀", text)

	require.Equal(t, []string{"你好", "呀"}, sink.userText, "running transcripts are sent as deltas")
	require.Equal(t, 2, session.pushed, "two audio frames, one stream")
}

func TestPipelineSendsStaticTools(t *testing.T) {
	llm := &fakeLLM{fragments: []string{"好的。"}}
	p := NewPipeline(Config{LLM: llm, Tools: []Tool{{FunctionDeclarations: []FunctionDeclaration{{Name: "book"}}}}})

	sink := &recordingSink{}
	require.NoError(t, p.Respond(context.Background(), "订会议室", sink))
	require.Len(t, llm.lastReq.Tools, 1, "Tools set at construction must reach the model")

	// A mid-session swap replaces them.
	require.NoError(t, p.UpdateTools(context.Background(), nil))
	require.NoError(t, p.Respond(context.Background(), "再说一次", sink))
	require.Empty(t, llm.lastReq.Tools)
}

// TestPipelinePassesProviderOptions pins the abstraction: every knob in Config
// reaches the stage it belongs to, instead of being baked into a provider.
func TestPipelinePassesProviderOptions(t *testing.T) {
	factory := &fakeASRFactory{}
	tts := &fakeTTS{}
	llm := &fakeLLM{fragments: []string{"好的。"}}

	asrOpts := ASROptions{Language: "zh-CN", SampleRate: 16000}
	ttsOpts := TTSOptions{Voice: "zh_female_x", Language: "zh", Model: "tts-model", SampleRate: 24000}
	p := NewPipeline(Config{
		ASR:        factory,
		LLM:        llm,
		TTS:        tts,
		ASROptions: asrOpts,
		TTSOptions: ttsOpts,
		LLMOptions: LLMOptions{
			Model:             "gemini-requested",
			SystemInstruction: "be brief",
			Temperature:       0.3,
			MaxOutputTokens:   64,
		},
	})

	ctx := context.Background()
	sink := &recordingSink{}
	require.NoError(t, p.PushAudio(ctx, []byte{1, 2}, sink))
	require.Equal(t, asrOpts, factory.lastOpts)

	require.NoError(t, p.Respond(ctx, "你好", sink))
	require.Equal(t, ttsOpts, tts.lastOpts)
	require.Equal(t, "gemini-requested", llm.lastReq.Model)
	require.Equal(t, "be brief", llm.lastReq.SystemInstruction)
	require.Equal(t, 0.3, llm.lastReq.Temperature)
	require.Equal(t, 64, llm.lastReq.MaxOutputTokens)
}

func TestPipelineASRReopensPerUtterance(t *testing.T) {
	factory := &fakeASRFactory{}
	p := NewPipeline(Config{ASR: factory})
	sink := &recordingSink{}
	ctx := context.Background()

	require.NoError(t, p.PushAudio(ctx, []byte{1}, sink))
	_, err := p.CommitUserTurn(ctx, sink)
	require.NoError(t, err)
	require.NoError(t, p.PushAudio(ctx, []byte{1}, sink))
	require.Equal(t, 2, factory.opened)
}

func TestBytePlusASRFraming(t *testing.T) {
	var (
		mu       sync.Mutex
		setup    map[string]any
		audio    int
		lastSeen bool
	)

	upgrader := websocket.Upgrader{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "test-key", r.Header.Get("X-Api-Key"))
		require.Equal(t, bytePlusASRDefaultResource, r.Header.Get("X-Api-Resource-Id"))
		require.NotEmpty(t, r.Header.Get("X-Api-Connect-Id"))

		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer func() { _ = conn.Close() }()

		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			frame, err := parseSAUC(data)
			require.NoError(t, err)
			require.Equal(t, byte(saucVersion), data[0])
			switch frame.messageType {
			case saucTypeFullReq:
				require.NotZero(t, frame.flags&saucFlagSequence, "the setup frame carries a sequence number")
				// The sequence field sits between the header and the size.
				seq := binary.BigEndian.Uint32(data[4:8])
				require.Equal(t, uint32(1), seq)
				size := binary.BigEndian.Uint32(data[8:12])
				require.NoError(t, json.Unmarshal(data[12:12+size], &setup))
				mu.Lock()
				mu.Unlock()
			case saucTypeAudio:
				mu.Lock()
				audio++
				done := frame.flags&saucFlagLast != 0
				lastSeen = done
				mu.Unlock()
				if len(frame.payload) > 0 {
					writeSAUCServerFrame(conn, 0x1, "你好")
				}
				if done {
					writeSAUCServerFrame(conn, 0x3, "你好呀")
					return
				}
			}
		}
	}))
	t.Cleanup(srv.Close)

	asr := &BytePlusASR{APIKey: "test-key", URL: "ws" + strings.TrimPrefix(srv.URL, "http")}
	var partials []string
	session, err := asr.Open(context.Background(), ASROptions{Language: "zh-CN"}, func(text string) error {
		partials = append(partials, text)
		return nil
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })

	require.NoError(t, session.Push(context.Background(), make([]byte, 3200)))
	require.NoError(t, session.Push(context.Background(), make([]byte, 3200)))

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	final, err := session.Finish(ctx)
	require.NoError(t, err)
	require.Equal(t, "你好呀", final)
	require.Equal(t, []string{"你好"}, partials)

	mu.Lock()
	require.Equal(t, 3, audio, "two audio frames plus the empty last-frame marker")
	require.True(t, lastSeen, "Finish must send a last-frame marker")
	mu.Unlock()

	require.Equal(t, "bigmodel", setup["request"].(map[string]any)["model_name"])
	require.Equal(t, "zh-CN", setup["request"].(map[string]any)["language"], "ASROptions.Language must reach the vendor request")
	audioCfg := setup["audio"].(map[string]any)
	require.Equal(t, "pcm", audioCfg["format"])
	require.Equal(t, float64(InputSampleRate), audioCfg["sample_rate"])
}

func TestBytePlusASRErrorFrame(t *testing.T) {
	upgrader := websocket.Upgrader{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer func() { _ = conn.Close() }()
		_, _, _ = conn.ReadMessage()
		payload := []byte(`{"error":"requested resource not granted"}`)
		frame := saucFrame(saucTypeError, 0, saucSerialization, 0, payload)
		_ = conn.WriteMessage(websocket.BinaryMessage, frame)
	}))
	t.Cleanup(srv.Close)

	asr := &BytePlusASR{APIKey: "k", URL: "ws" + strings.TrimPrefix(srv.URL, "http")}
	session, err := asr.Open(context.Background(), ASROptions{}, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err = session.Finish(ctx)
	require.ErrorContains(t, err, "resource not granted")
}

func TestBytePlusTTSSpeak(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		require.Equal(t, "test-key", r.Header.Get("x-api-key"))
		require.Equal(t, bytePlusTTSEndpoint, r.Header.Get("X-Api-Resource-Id"))
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
		// Two documents, like the real endpoint: the decoder must read them in
		// sequence rather than parse one blob.
		_, _ = io.WriteString(w, `{"code":0,"message":"","data":"AQID"}`)
		_, _ = io.WriteString(w, `{"code":0,"message":"","data":"BAUG"}`)
		// The real endpoint ends the stream with this document.
		_, _ = io.WriteString(w, `{"code":20000000,"message":"OK"}`)
	}))
	t.Cleanup(srv.Close)

	tts := &BytePlusTTS{APIKey: "test-key", URL: srv.URL, Speaker: "voice"}
	var frames [][]byte
	require.NoError(t, tts.Speak(context.Background(), "你好。", TTSOptions{}, func(pcm []byte) error {
		frames = append(frames, append([]byte(nil), pcm...))
		return nil
	}))

	require.Equal(t, "/", gotPath)
	require.Equal(t, [][]byte{{1, 2, 3}, {4, 5, 6}}, frames)
	params := gotBody["req_params"].(map[string]any)
	require.Equal(t, "你好。", params["text"])
	require.Equal(t, "voice", params["speaker"])
	audio := params["audio_params"].(map[string]any)
	require.Equal(t, "pcm", audio["format"])
	require.Equal(t, float64(OutputSampleRate), audio["sample_rate"])
}

// TestBytePlusTTSOptionOverrides pins the layering: per-call options win over
// the provider defaults, and the provider defaults win over the vendor default.
func TestBytePlusTTSOptionOverrides(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
		_, _ = io.WriteString(w, `{"code":20000000,"message":"OK"}`)
	}))
	t.Cleanup(srv.Close)

	tts := &BytePlusTTS{APIKey: "k", URL: srv.URL, Speaker: "provider-voice", Language: "en"}
	require.NoError(t, tts.Speak(context.Background(), "hi", TTSOptions{Voice: "call-voice", Language: "zh", Model: "m"}, func([]byte) error {
		return nil
	}))

	params := gotBody["req_params"].(map[string]any)
	require.Equal(t, "call-voice", params["speaker"])
	require.Equal(t, "zh", params["language"])
	require.Equal(t, "m", params["model"])

	// No per-call options: the provider's own values apply.
	require.NoError(t, tts.Speak(context.Background(), "hi", TTSOptions{}, func([]byte) error {
		return nil
	}))
	params = gotBody["req_params"].(map[string]any)
	require.Equal(t, "provider-voice", params["speaker"])
	require.Equal(t, "en", params["language"])
	require.NotContains(t, params, "model")
}

func TestBytePlusTTSErrorChunk(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"code":0,"data":"AQID"}`)
		_, _ = io.WriteString(w, `{"code":55000000,"message":"quota exceeded"}`)
	}))
	t.Cleanup(srv.Close)

	tts := &BytePlusTTS{APIKey: "k", URL: srv.URL}
	err := tts.Speak(context.Background(), "你好", TTSOptions{}, func([]byte) error { return nil })
	require.ErrorContains(t, err, "quota exceeded")
}

// --- fakes ---

type recordingSink struct {
	userText      []string
	assistantText []string
	audio         [][]byte
	toolCalls     []FunctionCall
	usageFrames   []UsageMetadata
	turnComplete  bool
	onAudio       func()
}

func (r *recordingSink) UserTranscription(_ context.Context, text string, _ bool) error {
	r.userText = append(r.userText, text)
	return nil
}

func (r *recordingSink) AssistantTranscription(_ context.Context, text string) error {
	r.assistantText = append(r.assistantText, text)
	return nil
}

func (r *recordingSink) Audio(ctx context.Context, pcm []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	r.audio = append(r.audio, pcm)
	if r.onAudio != nil {
		r.onAudio()
	}
	return nil
}

func (r *recordingSink) ToolCall(_ context.Context, calls []FunctionCall) error {
	r.toolCalls = append(r.toolCalls, calls...)
	return nil
}

func (r *recordingSink) Usage(_ context.Context, usage UsageMetadata) error {
	r.usageFrames = append(r.usageFrames, usage)
	return nil
}

func (r *recordingSink) TurnComplete(context.Context) error {
	r.turnComplete = true
	return nil
}

type fakeLLM struct {
	fragments []string
	toolCalls []FunctionCall
	lastReq   LLMRequest
}

func (f *fakeLLM) Stream(_ context.Context, req LLMRequest, onText func(string) error) (LLMResult, error) {
	f.lastReq = req
	for _, fragment := range f.fragments {
		if err := onText(fragment); err != nil {
			return LLMResult{}, err
		}
	}
	text := strings.Join(f.fragments, "")
	return LLMResult{
		Text:      text,
		ToolCalls: f.toolCalls,
		Usage:     UsageMetadata{PromptTokenCount: 3, CandidatesTokenCount: 2, TotalTokenCount: 5},
	}, nil
}

type fakeTTS struct {
	onSpeak  func(text string)
	err      error
	lastOpts TTSOptions
}

func (f *fakeTTS) Speak(_ context.Context, text string, opts TTSOptions, onPCM func([]byte) error) error {
	f.lastOpts = opts
	if f.onSpeak != nil {
		f.onSpeak(text)
	}
	if f.err != nil {
		return f.err
	}
	return onPCM([]byte{0, 0})
}

type fakeASRFactory struct {
	opened   int
	session  *fakeASRSession
	lastOpts ASROptions
}

func (f *fakeASRFactory) Open(_ context.Context, opts ASROptions, onPartial func(string) error) (ASRSessionHandle, error) {
	f.opened++
	f.lastOpts = opts
	f.session = &fakeASRSession{onPartial: onPartial}
	return f.session, nil
}

type fakeASRSession struct {
	onPartial func(string) error
	pushed    int
	closed    bool
}

func (s *fakeASRSession) Push(_ context.Context, pcm []byte) error {
	s.pushed++
	return nil
}

func (s *fakeASRSession) Finish(context.Context) (string, error) { return "你好呀", nil }

func (s *fakeASRSession) Close() error {
	s.closed = true
	return nil
}

func (s *fakeASRSession) partial(text string) error { return s.onPartial(text) }

func writeSAUCServerFrame(conn *websocket.Conn, flags byte, text string) {
	payload, _ := json.Marshal(map[string]any{
		"result": map[string]any{"text": text},
	})
	_ = conn.WriteMessage(websocket.BinaryMessage, saucFrame(saucTypeServer, flags, saucSerialization, 0, payload))
}
