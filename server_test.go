package realtimevoice

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

func newTestServer(t *testing.T, script Script, apiKey string) *httptest.Server {
	t.Helper()
	srv := &Server{
		APIKey: apiKey,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		Engine: func(context.Context, *Setup) (Engine, error) {
			return NewScriptedEngine(script), nil
		},
	}
	ts := httptest.NewServer(srv)
	t.Cleanup(ts.Close)
	return ts
}

func wsURL(ts *httptest.Server) string {
	return "ws" + strings.TrimPrefix(ts.URL, "http")
}

func dial(t *testing.T, ts *httptest.Server, query string, header http.Header) *websocket.Conn {
	t.Helper()
	c, _, err := websocket.DefaultDialer.Dial(wsURL(ts)+query, header)
	require.NoError(t, err)
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func setupSession(t *testing.T, c *websocket.Conn) {
	t.Helper()
	require.NoError(t, c.WriteJSON(ClientEvent{Setup: &Setup{Model: "models/gemini-3.8-live"}}))
	ev := readEvent(t, c)
	require.NotNil(t, ev.SetupComplete)
}

func readEvent(t *testing.T, c *websocket.Conn) ServerEvent {
	t.Helper()
	require.NoError(t, c.SetReadDeadline(time.Now().Add(5*time.Second)))
	var ev ServerEvent
	require.NoError(t, c.ReadJSON(&ev))
	return ev
}

func sendAudio(t *testing.T, c *websocket.Conn) {
	t.Helper()
	pcm := make([]byte, 640) // 20 ms at 16 kHz
	require.NoError(t, c.WriteJSON(ClientEvent{RealtimeInput: &RealtimeInput{
		MediaChunks: []Blob{{MimeType: InputMIMEType, Data: base64.StdEncoding.EncodeToString(pcm)}},
	}}))
}

func TestSessionAnswersAudioTurn(t *testing.T) {
	ts := newTestServer(t, Script{
		ChunkMS: 5,
		TTFBMS:  5,
		Turns:   []ScriptTurn{{UserTranscript: "你好", AssistantText: "你好呀", AudioMS: 60}},
	}, "")
	c := dial(t, ts, "", nil)
	setupSession(t, c)

	require.NoError(t, c.WriteJSON(ClientEvent{RealtimeInput: &RealtimeInput{ActivityStart: &struct{}{}}}))
	sendAudio(t, c)
	require.NoError(t, c.WriteJSON(ClientEvent{RealtimeInput: &RealtimeInput{ActivityEnd: &struct{}{}}}))

	var userText, assistantText strings.Builder
	audioFrames := 0
	var usage *UsageMetadata
	sawTurnComplete := false

	for !sawTurnComplete {
		ev := readEvent(t, c)
		if ev.UsageMetadata != nil {
			require.False(t, sawTurnComplete, "usage must arrive before turnComplete")
			usage = ev.UsageMetadata
			continue
		}
		sc := ev.ServerContent
		require.NotNil(t, sc, "unexpected frame: %+v", ev)
		if sc.InputTranscription != nil {
			userText.WriteString(sc.InputTranscription.Text)
		}
		if sc.OutputTranscription != nil {
			assistantText.WriteString(sc.OutputTranscription.Text)
		}
		if sc.ModelTurn != nil {
			audioFrames++
			require.Len(t, sc.ModelTurn.Parts, 1)
			blob := sc.ModelTurn.Parts[0].InlineData
			require.NotNil(t, blob)
			require.Equal(t, OutputMIMEType, blob.MimeType)
			pcm, err := blob.PCM()
			require.NoError(t, err)
			require.NotEmpty(t, pcm)
		}
		if sc.TurnComplete {
			sawTurnComplete = true
		}
	}

	require.Equal(t, "你好", userText.String())
	require.Equal(t, "你好呀", assistantText.String())
	require.Greater(t, audioFrames, 0)
	require.NotNil(t, usage)
	require.Greater(t, usage.PromptTokenCount, 0)
	require.Equal(t, usage.PromptTokenCount+usage.CandidatesTokenCount, usage.TotalTokenCount)
}

func TestBargeInStopsAudio(t *testing.T) {
	ts := newTestServer(t, Script{
		ChunkMS: 5,
		TTFBMS:  5,
		Turns:   []ScriptTurn{{UserTranscript: "u", AssistantText: "a long reply", AudioMS: 5000}},
	}, "")
	c := dial(t, ts, "", nil)
	setupSession(t, c)

	require.NoError(t, c.WriteJSON(ClientEvent{RealtimeInput: &RealtimeInput{ActivityStart: &struct{}{}}}))
	sendAudio(t, c)
	require.NoError(t, c.WriteJSON(ClientEvent{RealtimeInput: &RealtimeInput{ActivityEnd: &struct{}{}}}))

	// Wait for playback to start, then barge in.
	for {
		ev := readEvent(t, c)
		if ev.ServerContent != nil && ev.ServerContent.ModelTurn != nil {
			break
		}
	}
	require.NoError(t, c.WriteJSON(ClientEvent{RealtimeInput: &RealtimeInput{ActivityStart: &struct{}{}}}))

	for {
		ev := readEvent(t, c)
		if ev.ServerContent != nil && ev.ServerContent.Interrupted {
			break
		}
	}

	// Nothing else may be written for this turn: a late audio frame would
	// restart playback after the client was told to stop.
	require.NoError(t, c.SetReadDeadline(time.Now().Add(300*time.Millisecond)))
	for {
		_, _, err := c.ReadMessage()
		if err != nil {
			break
		}
		t.Fatalf("received a frame after interruption")
	}
}

func TestToolCallRoundTrip(t *testing.T) {
	ts := newTestServer(t, Script{
		ChunkMS: 5,
		TTFBMS:  5,
		Turns: []ScriptTurn{{
			UserTranscript: "book it",
			ToolCalls: []FunctionCall{{
				ID: "call-1", Name: "book", Args: json.RawMessage(`{"id":7}`),
			}},
			AfterToolText: "done",
			AfterToolMS:   40,
		}},
	}, "")
	c := dial(t, ts, "", nil)
	setupSession(t, c)

	require.NoError(t, c.WriteJSON(ClientEvent{RealtimeInput: &RealtimeInput{ActivityStart: &struct{}{}}}))
	sendAudio(t, c)
	require.NoError(t, c.WriteJSON(ClientEvent{RealtimeInput: &RealtimeInput{ActivityEnd: &struct{}{}}}))

	var call *ServerToolCall
	for call == nil {
		ev := readEvent(t, c)
		if sc := ev.ServerContent; sc != nil {
			// The user's transcription is expected; model audio is not, because
			// the turn stops at the tool call.
			require.Nil(t, sc.ModelTurn, "no audio before the tool call")
			require.Nil(t, sc.OutputTranscription, "no assistant text before the tool call")
			require.False(t, sc.TurnComplete, "a tool call is not a turn boundary")
		}
		call = ev.ToolCall
	}
	require.Len(t, call.FunctionCalls, 1)
	require.Equal(t, "book", call.FunctionCalls[0].Name)
	require.JSONEq(t, `{"id":7}`, string(call.FunctionCalls[0].Args))

	require.NoError(t, c.WriteJSON(ClientEvent{ToolResponse: &ToolResponse{
		FunctionResponses: []FunctionResponse{{
			ID: "call-1", Name: "book", Response: json.RawMessage(`{"ok":true}`),
		}},
	}}))

	var text strings.Builder
	audioFrames := 0
	for {
		ev := readEvent(t, c)
		if ev.ServerContent == nil {
			continue
		}
		if ev.ServerContent.OutputTranscription != nil {
			text.WriteString(ev.ServerContent.OutputTranscription.Text)
		}
		if ev.ServerContent.ModelTurn != nil {
			audioFrames++
		}
		if ev.ServerContent.TurnComplete {
			break
		}
	}
	require.Equal(t, "done", text.String())
	require.Greater(t, audioFrames, 0)
}

func TestTextTurn(t *testing.T) {
	ts := newTestServer(t, Script{
		ChunkMS: 5,
		TTFBMS:  5,
		Turns:   []ScriptTurn{{AssistantText: "hi", AudioMS: 20}},
	}, "")
	c := dial(t, ts, "", nil)
	setupSession(t, c)

	require.NoError(t, c.WriteJSON(ClientEvent{ClientContent: &ClientContent{
		Turns:        []Content{{Role: "user", Parts: []Part{{Text: "hello"}}}},
		TurnComplete: true,
	}}))

	for {
		ev := readEvent(t, c)
		if ev.UsageMetadata != nil {
			continue
		}
		require.NotNil(t, ev.ServerContent)
		if ev.ServerContent.TurnComplete {
			return
		}
	}
}

func TestAuth(t *testing.T) {
	ts := newTestServer(t, Script{}, "s3cret")

	_, _, err := websocket.DefaultDialer.Dial(wsURL(ts), nil)
	require.Error(t, err, "missing key must be rejected")

	_, _, err = websocket.DefaultDialer.Dial(wsURL(ts)+"?key=wrong", nil)
	require.Error(t, err)

	c := dial(t, ts, "?key=s3cret", nil)
	setupSession(t, c)

	c = dial(t, ts, "", http.Header{"Authorization": []string{"Bearer s3cret"}})
	setupSession(t, c)
}

func TestSetupMustBeFirst(t *testing.T) {
	ts := newTestServer(t, Script{}, "")
	c := dial(t, ts, "", nil)

	require.NoError(t, c.WriteJSON(ClientEvent{RealtimeInput: &RealtimeInput{ActivityStart: &struct{}{}}}))
	require.NoError(t, c.SetReadDeadline(time.Now().Add(2*time.Second)))
	_, _, err := c.ReadMessage()
	require.Error(t, err, "a frame before setup must close the session")
}

func TestWireShapes(t *testing.T) {
	t.Run("server audio frame", func(t *testing.T) {
		data, err := json.Marshal(ServerEvent{ServerContent: &ServerContent{
			ModelTurn: NewAudioContent([]byte{1, 2}),
		}})
		require.NoError(t, err)
		require.JSONEq(t,
			`{"serverContent":{"modelTurn":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AQI="}}]}}}`,
			string(data))
	})

	t.Run("legacy and current realtime input", func(t *testing.T) {
		var legacy ClientEvent
		require.NoError(t, json.Unmarshal([]byte(
			`{"realtimeInput":{"mediaChunks":[{"mimeType":"audio/pcm;rate=16000","data":"AAAA"}]}}`), &legacy))
		require.Len(t, legacy.RealtimeInput.MediaChunks, 1)

		var current ClientEvent
		require.NoError(t, json.Unmarshal([]byte(
			`{"realtimeInput":{"audio":{"mimeType":"audio/pcm;rate=16000","data":"AAAA"},"activityStart":{},"audioStreamEnd":true}}`), &current))
		require.NotNil(t, current.RealtimeInput.Audio)
		require.NotNil(t, current.RealtimeInput.ActivityStart)
		require.True(t, current.RealtimeInput.AudioStreamEnd)
	})

	t.Run("setup with tools and transcription", func(t *testing.T) {
		var ev ClientEvent
		require.NoError(t, json.Unmarshal([]byte(`{"setup":{
			"model":"models/gemini-3.8-live",
			"systemInstruction":{"parts":[{"text":"be brief"}]},
			"tools":[{"functionDeclarations":[{"name":"book","parameters":{"type":"object"}}]}],
			"inputAudioTranscription":{"languageCodes":["zh-CN","en"]},
			"outputAudioTranscription":{},
			"generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Puck"}}}}
		}}`), &ev))
		require.Equal(t, "models/gemini-3.8-live", ev.Setup.Model)
		require.Equal(t, "be brief", ev.Setup.SystemInstruction.Parts[0].Text)
		require.Equal(t, "book", ev.Setup.Tools[0].FunctionDeclarations[0].Name)
		require.Equal(t, []string{"zh-CN", "en"}, ev.Setup.InputAudioTranscription.LanguageCodes)
		require.Equal(t, "Puck",
			ev.Setup.GenerationConfig.SpeechConfig.VoiceConfig.PrebuiltVoiceConfig.VoiceName)
	})
}
