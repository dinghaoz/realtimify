package geminilive

import "context"

// The pipeline is three swappable stages. Each stage receives its own options
// struct on every call, so a running server can vary the voice, the language or
// the model per session without rebuilding the pipeline.

// ASR turns microphone audio into text. Open is called once per spoken
// utterance; the returned session is fed the whole utterance and closed at its
// end.
type ASR interface {
	// Open starts one recognition stream. onPartial receives running
	// transcripts — the whole current hypothesis, not a delta — while the user
	// is still speaking; an empty callback is allowed for batch providers.
	Open(ctx context.Context, opts ASROptions, onPartial func(text string) error) (ASRSessionHandle, error)
}

// ASROptions configures one recognition stream.
type ASROptions struct {
	// Language is a hint such as "zh-CN" or "en-US"; empty means auto-detect
	// where the provider supports it.
	Language string
	// SampleRate is the PCM rate of the pushed audio. Zero means the provider's
	// default, which is 16000 for every implementation shipped here.
	SampleRate int
}

// ASRSessionHandle is the small surface the pipeline needs from a recognizer.
type ASRSessionHandle interface {
	Push(ctx context.Context, pcm16k []byte) error
	// Finish ends the stream and returns the final transcript.
	Finish(ctx context.Context) (string, error)
	Close() error
}

// TTS synthesizes one clause into audio.
type TTS interface {
	// Speak streams audio for text. onPCM is called with successive PCM
	// fragments; returning an error stops synthesis (barge-in).
	Speak(ctx context.Context, text string, opts TTSOptions, onPCM func(pcm []byte) error) error
}

// TTSOptions configures one synthesis call.
type TTSOptions struct {
	// Voice is the provider's voice identifier (音色). Empty uses the
	// implementation default.
	Voice string
	// Language is an optional hint such as "zh" or "en".
	Language string
	// Model optionally selects the synthesis model.
	Model string
	// SampleRate is the PCM rate handed to onPCM. Zero means the provider's
	// default, which is 24000 for every implementation shipped here.
	SampleRate int
}

// LLMOptions configures the text-generation stage.
type LLMOptions struct {
	// Model selects the text model; empty uses the provider default.
	Model string
	// SystemInstruction is prepended to every turn.
	SystemInstruction string
	Temperature       float64
	MaxOutputTokens   int
}

// Config wires the three stages and their per-call options.
//
// Only LLM is required for a text-only session; TTS may be nil to skip audio,
// and ASR may be nil when the client sends text through clientContent.
type Config struct {
	ASR ASR
	TTS TTS
	LLM TextLLM

	ASROptions ASROptions
	TTSOptions TTSOptions
	LLMOptions LLMOptions

	// Tools are the function declarations offered to the model on every turn
	// until UpdateTools replaces them.
	Tools []Tool

	// MaxClauseRunes splits long replies so the first audio frame does not wait
	// for a whole paragraph. Zero means 40 runes.
	MaxClauseRunes int
}
