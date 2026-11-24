package geminilive

import (
	"context"
	"encoding/binary"
	"math"
	"sync"
	"time"
)

// Audio wire formats. The Live API feeds 16 kHz PCM in and emits 24 kHz PCM
// out; both are signed 16-bit little-endian mono.
const (
	InputMIMEType    = "audio/pcm;rate=16000"
	OutputMIMEType   = "audio/pcm;rate=24000"
	InputSampleRate  = 16000
	OutputSampleRate = 24000
)

// Sink receives the protocol-level output of one assistant turn. Implementations
// must be safe for concurrent use: audio can be pushed from the reader
// goroutine while the turn runs on its own goroutine.
//
// Errors are returned so an engine stops promptly when the socket dies; a
// cancelled context means the turn was interrupted and the engine must stop
// emitting.
type Sink interface {
	// UserTranscription reports the user's speech. final=false is a partial,
	// final=true commits the utterance line.
	UserTranscription(ctx context.Context, text string, final bool) error
	// AssistantTranscription streams the assistant's words (what the TTS is
	// speaking), in ordered fragments.
	AssistantTranscription(ctx context.Context, text string) error
	// Audio streams 24 kHz 16-bit mono PCM.
	Audio(ctx context.Context, pcm24k []byte) error
	// ToolCall asks the client to run functions. Per the Live API, no
	// TurnComplete follows a tool call; the turn resumes on ToolResponse.
	ToolCall(ctx context.Context, calls []FunctionCall) error
	// Usage reports token estimates for the turn. It is emitted before
	// TurnComplete so a client that bills at the turn boundary catches it.
	Usage(ctx context.Context, usage UsageMetadata) error
	// TurnComplete closes the assistant turn.
	TurnComplete(ctx context.Context) error
}

// Engine is the speech/LLM backend behind the protocol. The server owns the
// socket and the frame shapes; the engine owns recognition, generation and
// synthesis.
//
// Concurrency contract: PushAudio runs on the socket reader goroutine, while
// Respond / SubmitToolResults run on a turn goroutine. An implementation must
// tolerate them overlapping, and must stop as soon as ctx is cancelled.
type Engine interface {
	// PushAudio feeds microphone PCM normalized to InputSampleRate. sink is the
	// session-level sink so a streaming recognizer can report partial
	// transcription while the user is still speaking.
	PushAudio(ctx context.Context, pcm16k []byte, sink Sink) error
	// CommitUserTurn finalizes the pending utterance and returns its transcript.
	// An empty transcript means no speech was detected and the server should not
	// start a response.
	CommitUserTurn(ctx context.Context, sink Sink) (string, error)
	// Respond produces one assistant turn for userText.
	Respond(ctx context.Context, userText string, sink Sink) error
	// SubmitToolResults resumes a turn after the client ran its functions.
	SubmitToolResults(ctx context.Context, results []FunctionResponse, sink Sink) error
	// UpdateTools applies a mid-session context update. Engines that cannot swap
	// tools in place should return ErrToolUpdateUnsupported.
	UpdateTools(ctx context.Context, tools []Tool) error
	// Close releases the engine's resources. It must be idempotent.
	Close() error
}

// ErrToolUpdateUnsupported is returned by an engine that cannot change its
// tool set mid-session. The server then leaves the session running; callers
// that need the swap should reconnect.
var ErrToolUpdateUnsupported = errString("geminilive: tool update unsupported")

type errString string

func (e errString) Error() string { return string(e) }

// ScriptTurn is one canned exchange for ScriptedEngine.
type ScriptTurn struct {
	// UserTranscript is what the next committed utterance is reported to be.
	UserTranscript string
	// AssistantText is streamed as outputTranscription fragments.
	AssistantText string
	// AudioMS is how much synthesized audio to emit for the reply.
	AudioMS int
	// ToolCalls, when present, are emitted instead of finishing the turn; the
	// turn finishes on ToolResponse with AfterTool*.
	ToolCalls     []FunctionCall
	AfterToolText string
	AfterToolMS   int
}

// Script is the canned behavior of a ScriptedEngine. It exists so the protocol
// can be exercised end to end without speech vendors: it says exactly what the
// script says, no more.
type Script struct {
	Turns []ScriptTurn
	// Loop replays the script after the last turn instead of repeating the last.
	Loop bool
	// ChunkMS is the audio frame size (default 20 ms).
	ChunkMS int
	// TTFBMS delays the first assistant output (default 250 ms), so barge-in
	// tests have a window to interrupt.
	TTFBMS int
}

// ScriptedEngine is a deterministic Engine driven by a Script. It is the
// conformance backend for the protocol server and the placeholder for a real
// STT+LLM+TTS pipeline.
type ScriptedEngine struct {
	script Script

	mu         sync.Mutex
	next       int
	audioBytes int
	tools      []Tool
	// turn is the script turn the current exchange is committed to. It is taken
	// once (on CommitUserTurn, or on a text-only Respond) so the transcript, the
	// reply and the post-tool continuation all come from the same script entry.
	turn   ScriptTurn
	inTurn bool
}

// NewScriptedEngine builds a ScriptedEngine. Zero-valued timing fields take
// their defaults.
func NewScriptedEngine(s Script) *ScriptedEngine {
	if s.ChunkMS <= 0 {
		s.ChunkMS = 20
	}
	if s.TTFBMS <= 0 {
		s.TTFBMS = 250
	}
	if len(s.Turns) == 0 {
		s.Turns = []ScriptTurn{{UserTranscript: "hello", AssistantText: "hello there", AudioMS: 500}}
	}
	return &ScriptedEngine{script: s}
}

// PushAudio records the input size (for the token estimate) and otherwise
// waits for the turn to be committed.
func (e *ScriptedEngine) PushAudio(_ context.Context, pcm16k []byte, _ Sink) error {
	e.mu.Lock()
	e.audioBytes += len(pcm16k)
	e.mu.Unlock()
	return nil
}

// CommitUserTurn advances the script and reports the user's line as final.
func (e *ScriptedEngine) CommitUserTurn(ctx context.Context, sink Sink) (string, error) {
	e.mu.Lock()
	e.turn = e.current()
	e.inTurn = true
	text := e.turn.UserTranscript
	e.mu.Unlock()

	if text == "" {
		return "", nil
	}
	if err := sink.UserTranscription(ctx, text, true); err != nil {
		return "", err
	}
	return text, nil
}

// Respond streams the scripted assistant turn.
func (e *ScriptedEngine) Respond(ctx context.Context, userText string, sink Sink) error {
	e.mu.Lock()
	if !e.inTurn {
		e.turn = e.current()
		e.inTurn = true
	}
	turn := e.turn
	e.mu.Unlock()

	if err := sleepCtx(ctx, time.Duration(e.script.TTFBMS)*time.Millisecond); err != nil {
		return err
	}
	if turn.AssistantText != "" {
		if err := streamText(ctx, sink, turn.AssistantText, e.script.ChunkMS); err != nil {
			return err
		}
	}
	if len(turn.ToolCalls) > 0 {
		return sink.ToolCall(ctx, turn.ToolCalls)
	}
	if err := e.streamAudio(ctx, sink, turn.AudioMS); err != nil {
		return err
	}
	if err := e.finish(ctx, sink, userText, turn.AssistantText, turn.AudioMS); err != nil {
		return err
	}
	e.endTurn()
	return nil
}

// SubmitToolResults finishes the turn the tool call opened.
func (e *ScriptedEngine) SubmitToolResults(ctx context.Context, _ []FunctionResponse, sink Sink) error {
	e.mu.Lock()
	if !e.inTurn {
		e.turn = e.current()
		e.inTurn = true
	}
	turn := e.turn
	e.mu.Unlock()

	if turn.AfterToolText != "" {
		if err := streamText(ctx, sink, turn.AfterToolText, e.script.ChunkMS); err != nil {
			return err
		}
	}
	if err := e.streamAudio(ctx, sink, turn.AfterToolMS); err != nil {
		return err
	}
	if err := e.finish(ctx, sink, turn.UserTranscript, turn.AfterToolText, turn.AfterToolMS); err != nil {
		return err
	}
	e.endTurn()
	return nil
}

// UpdateTools records the new tool set; the script does not use it.
func (e *ScriptedEngine) UpdateTools(_ context.Context, tools []Tool) error {
	e.mu.Lock()
	e.tools = tools
	e.mu.Unlock()
	return nil
}

// Close is a no-op; the scripted engine owns no resources.
func (e *ScriptedEngine) Close() error { return nil }

// current returns the turn the script is on and advances to the next one.
func (e *ScriptedEngine) current() ScriptTurn {
	if e.next >= len(e.script.Turns) && !e.script.Loop {
		return e.script.Turns[len(e.script.Turns)-1]
	}
	t := e.script.Turns[e.next%len(e.script.Turns)]
	e.next++
	return t
}

func (e *ScriptedEngine) streamAudio(ctx context.Context, sink Sink, ms int) error {
	if ms <= 0 {
		return nil
	}
	const chunkSamples = 480 // 20 ms at 24 kHz
	samplesPerMS := OutputSampleRate / 1000
	total := ms * samplesPerMS
	for done := 0; done < total; done += chunkSamples {
		n := chunkSamples
		if total-done < n {
			n = total - done
		}
		if err := sleepCtx(ctx, time.Duration(e.script.ChunkMS)*time.Millisecond); err != nil {
			return err
		}
		if err := sink.Audio(ctx, tone24k(done, n)); err != nil {
			return err
		}
	}
	return nil
}

func (e *ScriptedEngine) endTurn() {
	e.mu.Lock()
	e.inTurn = false
	e.mu.Unlock()
}

func (e *ScriptedEngine) finish(ctx context.Context, sink Sink, userText, assistantText string, audioMS int) error {
	e.mu.Lock()
	inAudio := e.audioBytes / 1000 // ~1000 bytes per audio token at 16 kHz
	e.audioBytes = 0
	e.mu.Unlock()

	usage := UsageMetadata{
		PromptTokenCount:     inAudio + runeLen(userText),
		CandidatesTokenCount: audioMS*OutputSampleRate/1000/40 + runeLen(assistantText),
	}
	usage.PromptTokensDetails = []TokenCount{
		{Modality: "AUDIO", TokenCount: inAudio},
		{Modality: "TEXT", TokenCount: runeLen(userText)},
	}
	usage.CandidateTokenDetails = []TokenCount{
		{Modality: "AUDIO", TokenCount: usage.CandidatesTokenCount - runeLen(assistantText)},
		{Modality: "TEXT", TokenCount: runeLen(assistantText)},
	}
	usage.TotalTokenCount = usage.PromptTokenCount + usage.CandidatesTokenCount

	if err := sink.Usage(ctx, usage); err != nil {
		return err
	}
	return sink.TurnComplete(ctx)
}

// streamText emits text in small fragments to mimic incremental generation.
func streamText(ctx context.Context, sink Sink, text string, delayMS int) error {
	runes := []rune(text)
	const frag = 8
	for i := 0; i < len(runes); i += frag {
		end := i + frag
		if end > len(runes) {
			end = len(runes)
		}
		if err := sink.AssistantTranscription(ctx, string(runes[i:end])); err != nil {
			return err
		}
		if err := sleepCtx(ctx, time.Duration(delayMS)*time.Millisecond); err != nil {
			return err
		}
	}
	return nil
}

// tone24k renders n samples of a 440 Hz sine starting at absolute sample
// offset start. It is deterministic filler so clients can measure playback
// and barge-in without a TTS vendor.
func tone24k(start, n int) []byte {
	out := make([]byte, n*2)
	for i := 0; i < n; i++ {
		s := math.Sin(2 * math.Pi * 440 * float64(start+i) / OutputSampleRate)
		binary.LittleEndian.PutUint16(out[i*2:], uint16(int16(s*0.2*math.MaxInt16)))
	}
	return out
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func runeLen(s string) int {
	n := 0
	for range s {
		n++
	}
	return n
}
