# realtime-voice

A server that speaks the **Gemini Live** (`BidiGenerateContent`) WebSocket
protocol, backed by whatever speech stack you plug in.

A client written against Gemini Live connects unchanged — `setup`,
`realtimeInput` audio chunks, `activityStart`/`activityEnd`, `clientContent`,
`toolResponse`, barge-in, transcriptions — and the server answers with
`serverContent` frames, 24 kHz PCM audio and `turnComplete`.

Internally it is three swappable stages:

```
microphone PCM ──► ASR ──► text LLM ──► TTS ──► 24 kHz PCM
                    ▲          ▲          ▲
                    └── Config.ASROptions / LLMOptions / TTSOptions
```

Nothing in here is tied to a business domain, and no stage is tied to a vendor:
`ASR`, `TextLLM` and `TTS` are interfaces.

## Wiring

```go
pipeline := realtimevoice.NewPipeline(realtimevoice.Config{
	ASR: myASR,   // realtimevoice.ASR
	LLM: myLLM,   // realtimevoice.TextLLM
	TTS: myTTS,   // realtimevoice.TTS

	ASROptions: realtimevoice.ASROptions{Language: "zh-CN", SampleRate: 16000},
	TTSOptions: realtimevoice.TTSOptions{Voice: "zh_female_vv_uranus_bigtts", SampleRate: 24000},
	LLMOptions: realtimevoice.LLMOptions{
		Model:             "gemini-2.5-flash",
		SystemInstruction: "You are a voice assistant. Answer in one short sentence.",
		Temperature:       0.6,
	},
	Tools:          setup.Tools, // function declarations from the client's setup frame
	MaxClauseRunes: 40,          // split long replies so audio starts early
})

srv := &realtimevoice.Server{
	APIKey: "client-facing-key",
	Engine: func(ctx context.Context, setup *realtimevoice.Setup) (realtimevoice.Engine, error) {
		return pipeline, nil
	},
}
```

### The three stages

| Stage | Interface | Per-call options |
|-------|-----------|------------------|
| Speech to text | `ASR.Open(ctx, ASROptions, onPartial)` → `ASRSessionHandle` | language, sample rate |
| Text generation | `TextLLM.Stream(ctx, LLMRequest, onText)` → `LLMResult` | model, system instruction, temperature, max tokens, tools |
| Speech synthesis | `TTS.Speak(ctx, text, TTSOptions, onPCM)` | voice (音色), language, model, sample rate |

Options are passed on **every call**, not baked into the provider, so one
server can serve sessions in different languages or voices. Providers keep
their own defaults; a per-call value wins over the provider field, which in turn
wins over the vendor default.

A provider is ~20 lines. Speech synthesis, for example:

```go
type MyTTS struct{ apiKey string }

func (t *MyTTS) Speak(ctx context.Context, text string, opts realtimevoice.TTSOptions, onPCM func([]byte) error) error {
	// ... synthesize text with opts.Voice / opts.Language / opts.SampleRate,
	// calling onPCM for each fragment. Return ctx.Err() to stop on barge-in.
	return nil
}
```

The turn loop is provider-agnostic:

- the LLM stream is split at punctuation, and each clause is spoken as soon as
  it completes — the first audio frame does not wait for the whole reply;
- a tool call is emitted as `toolCall` and **does not** complete the turn; the
  client's `toolResponse` resumes the same turn, and the call/response pair is
  replayed to the model as history;
- `activityStart` during playback emits `interrupted` and stops synthesis;
- running ASR transcripts are reported as `inputTranscription` deltas while the
  user is still speaking.

## Running the reference server

`cmd/voice-server` ships two engines:

- **scripted** (default) — no credentials, deterministic audio: a protocol
  conformance harness;
- **live** — BytePlus Seed Speech (ASR + TTS) and Gemini (text). Enabled when
  `-llm-key` and `-asr-key` are set.

```bash
go run ./cmd/voice-server \
  -api-key devkey \
  -llm-key "$VERTEX_API_KEY" -llm-project "$VERTEX_PROJECT_ID" -llm-model gemini-2.5-flash \
  -asr-key "$BYTEPLUS_API_KEY" \
  -voice zh_female_vv_uranus_bigtts \
  -asr-language zh-CN \
  -temperature 0.6 \
  -system "你是一个实时语音助手。回答口语化、简短，最多两句话。"
```

Every knob has a flag: `-voice` (音色), `-asr-language`, `-tts-language`,
`-tts-model`, `-llm-model`, `-temperature`, `-max-tokens`, `-clause-runes`,
`-system`. None of them are logged; keys travel in headers only.

Connect to the same paths Gemini uses:

```
/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent        # Vertex
/ws/google.ai.generativelanguage.v1beta1.GenerativeService/BidiGenerateContent # AI Studio
```

### Measured latency

Voice in → first audio out, one turn, `gemini-2.5-flash` + BytePlus Seed Speech
from Singapore:

| Stage | Time |
|-------|------|
| ASR final after `activityEnd` | ~0.2 s |
| LLM first token | 1.2–1.5 s |
| TTS first audio | ~0.4 s |
| **First audio after the user stops talking** | **1.7–2.3 s** |
| Full turn | 2.0–2.6 s |

The LLM dominates. Everything else already streams.

## Testing

```bash
go test -race ./...
```

The suite covers the wire protocol, SAUC framing, TTS stream decoding, clause
splitting, barge-in, tool round trips and option pass-through, all without
network access.

Live vendor tests are behind a build tag and skip themselves without
credentials:

```bash
VERTEX_API_KEY=... VERTEX_PROJECT_ID=... go test -tags livellm ./... -run Live -v
```

## Layout

```
wire.go            Gemini Live wire types
server.go          WebSocket session: setup, input, sink → frames
engine.go          Engine/Sink interfaces + scripted engine
config.go          Config and the three stage interfaces with their options
pipeline.go        ASR → LLM → TTS turn loop
llm.go             Gemini text LLM
byteplus_asr.go    BytePlus Seed ASR 2.0 (SAUC binary framing)
byteplus_tts.go    BytePlus Seed TTS 2.0
cmd/voice-server
```
