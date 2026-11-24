// Package realtimevoice implements the server side of the Gemini Live
// (BidiGenerateContent) WebSocket protocol.
//
// The frame types below mirror the production client in
// service/llm/provider/gemini_live.go so a client written against Gemini can
// point at this server unchanged: same envelope fields, same snake/camel
// casing, same transcription and tool-call semantics. Deliberate divergences
// are documented where they occur (see ServerContent.GenerationComplete and
// the absence of an error envelope).
package realtimevoice

import (
	"encoding/base64"
	"encoding/json"
)

// --- client -> server ---

// ClientEvent is one frame sent by the client. Exactly one field is set.
type ClientEvent struct {
	Setup         *Setup         `json:"setup,omitempty"`
	RealtimeInput *RealtimeInput `json:"realtimeInput,omitempty"`
	ClientContent *ClientContent `json:"clientContent,omitempty"`
	ToolResponse  *ToolResponse  `json:"toolResponse,omitempty"`
	ContextUpdate *ContextUpdate `json:"contextUpdate,omitempty"`
}

// Setup is the first frame of a session. Fields this server does not act on
// (contextWindowCompression, responseModalities) are accepted and ignored so a
// real Gemini client's setup is not rejected.
type Setup struct {
	Model                    string                   `json:"model"`
	GenerationConfig         *GenerationConfig        `json:"generationConfig,omitempty"`
	SystemInstruction        *Content                 `json:"systemInstruction,omitempty"`
	Tools                    []Tool                   `json:"tools,omitempty"`
	InputAudioTranscription  *TranscriptionConfig     `json:"inputAudioTranscription,omitempty"`
	OutputAudioTranscription *TranscriptionConfig     `json:"outputAudioTranscription,omitempty"`
	SessionResumption        *SessionResumptionConfig `json:"sessionResumption,omitempty"`
	ContextWindowCompression json.RawMessage          `json:"contextWindowCompression,omitempty"`
}

// GenerationConfig carries the response modality and voice selection.
type GenerationConfig struct {
	ResponseModalities []string      `json:"responseModalities,omitempty"`
	SpeechConfig       *SpeechConfig `json:"speechConfig,omitempty"`
}

// SpeechConfig selects a prebuilt voice.
type SpeechConfig struct {
	VoiceConfig *VoiceConfig `json:"voiceConfig,omitempty"`
}

// VoiceConfig wraps the prebuilt voice selection.
type VoiceConfig struct {
	PrebuiltVoiceConfig *PrebuiltVoice `json:"prebuiltVoiceConfig,omitempty"`
}

// PrebuiltVoice names a Gemini Live voice. A synthetic pipeline maps this onto
// its own TTS voice set; the name is otherwise opaque.
type PrebuiltVoice struct {
	VoiceName string `json:"voiceName,omitempty"`
}

// TranscriptionConfig turns on transcription for one side. Presence alone
// enables it, matching the Live API.
type TranscriptionConfig struct {
	LanguageCodes []string `json:"languageCodes,omitempty"`
}

// SessionResumptionConfig asks for resumption handles.
type SessionResumptionConfig struct {
	Handle      string `json:"handle,omitempty"`
	Transparent bool   `json:"transparent,omitempty"`
}

// RealtimeInput streams microphone audio. The Live API has had two shapes:
// the legacy repeated mediaChunks and the current audio field. Both are
// accepted; activityStart/activityEnd bound the user's utterance and
// audioStreamEnd is the AudioStreamEnd alias.
type RealtimeInput struct {
	MediaChunks    []Blob    `json:"mediaChunks,omitempty"`
	Audio          *Blob     `json:"audio,omitempty"`
	Text           string    `json:"text,omitempty"`
	ActivityStart  *struct{} `json:"activityStart,omitempty"`
	ActivityEnd    *struct{} `json:"activityEnd,omitempty"`
	AudioStreamEnd bool      `json:"audioStreamEnd,omitempty"`
}

// Blob is inline binary data (base64 in JSON).
type Blob struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

// PCM decodes the base64 payload.
func (b Blob) PCM() ([]byte, error) {
	return base64.StdEncoding.DecodeString(b.Data)
}

// ClientContent sends typed turns (text history or a typed user message).
type ClientContent struct {
	Turns        []Content `json:"turns,omitempty"`
	TurnComplete bool      `json:"turnComplete,omitempty"`
}

// Content is a conversation turn.
type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts,omitempty"`
}

// Part is one piece of a turn: text, inline audio, or a function call /
// response.
type Part struct {
	Text             string            `json:"text,omitempty"`
	InlineData       *Blob             `json:"inlineData,omitempty"`
	FunctionCall     *FunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *FunctionResponse `json:"functionResponse,omitempty"`
}

// Tool is a group of function declarations.
type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations,omitempty"`
}

// FunctionDeclaration describes one callable function.
type FunctionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// FunctionCall is the model asking the client to run a function. Args is kept
// as raw JSON so the server never re-encodes a caller's payload.
type FunctionCall struct {
	ID   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

// FunctionResponse is the client's answer to a FunctionCall.
type FunctionResponse struct {
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response,omitempty"`
}

// ToolResponse carries one or more function responses. Gemini requires the
// whole batch from a single toolCall frame to come back in ONE toolResponse,
// so this is a repeated field and not a per-call frame.
type ToolResponse struct {
	FunctionResponses []FunctionResponse `json:"functionResponses,omitempty"`
}

// ContextUpdate replaces tools and/or the system instruction mid-session
// (BidiGenerateContentContextUpdate, Gemini 3.8 Live).
type ContextUpdate struct {
	SystemInstruction *Content      `json:"system_instruction,omitempty"`
	Tools             *ToolsWrapper `json:"tools,omitempty"`
}

// ToolsWrapper adds a presence bit: a bare repeated field cannot distinguish
// "no update" from "clear all tools".
type ToolsWrapper struct {
	Tools []Tool `json:"tools"`
}

// --- server -> client ---

// ServerEvent is one frame sent to the client. Exactly one field is set.
type ServerEvent struct {
	SetupComplete           *struct{}                `json:"setupComplete,omitempty"`
	ServerContent           *ServerContent           `json:"serverContent,omitempty"`
	ToolCall                *ServerToolCall          `json:"toolCall,omitempty"`
	UsageMetadata           *UsageMetadata           `json:"usageMetadata,omitempty"`
	GoAway                  *GoAway                  `json:"goAway,omitempty"`
	SessionResumptionUpdate *SessionResumptionUpdate `json:"sessionResumptionUpdate,omitempty"`
}

// ServerContent is the streaming model output envelope. ModelTurn audio is
// 24 kHz 16-bit mono PCM, matching the Live API.
type ServerContent struct {
	ModelTurn           *Content       `json:"modelTurn,omitempty"`
	TurnComplete        bool           `json:"turnComplete,omitempty"`
	Interrupted         bool           `json:"interrupted,omitempty"`
	GenerationComplete  bool           `json:"generationComplete,omitempty"`
	InputTranscription  *Transcription `json:"inputTranscription,omitempty"`
	OutputTranscription *Transcription `json:"outputTranscription,omitempty"`
}

// Transcription is the next fragment of an utterance, not the running total.
type Transcription struct {
	Text string `json:"text,omitempty"`
}

// ServerToolCall is the top-level tool-call frame. Live tool calls are NOT
// carried inside serverContent.modelTurn.parts.
type ServerToolCall struct {
	FunctionCalls []FunctionCall `json:"functionCalls,omitempty"`
}

// UsageMetadata reports per-turn token counts. A synthetic pipeline has no
// vendor tokenizer, so its numbers are estimates (documented by the engine).
type UsageMetadata struct {
	PromptTokenCount      int          `json:"promptTokenCount"`
	CandidatesTokenCount  int          `json:"candidatesTokenCount"`
	TotalTokenCount       int          `json:"totalTokenCount"`
	PromptTokensDetails   []TokenCount `json:"promptTokensDetails,omitempty"`
	CandidateTokenDetails []TokenCount `json:"candidatesTokensDetails,omitempty"`
}

// TokenCount is one modality's share of a token total.
type TokenCount struct {
	Modality   string `json:"modality"`
	TokenCount int    `json:"tokenCount"`
}

// GoAway precedes a server-initiated close.
type GoAway struct {
	TimeLeft string `json:"timeLeft,omitempty"`
}

// SessionResumptionUpdate carries a reconnect handle.
type SessionResumptionUpdate struct {
	NewHandle string `json:"newHandle,omitempty"`
	Resumable bool   `json:"resumable,omitempty"`
}

// NewAudioContent builds a modelTurn carrying one chunk of 24 kHz PCM.
func NewAudioContent(pcm []byte) *Content {
	return &Content{
		Role: "model",
		Parts: []Part{{
			InlineData: &Blob{
				MimeType: OutputMIMEType,
				Data:     base64.StdEncoding.EncodeToString(pcm),
			},
		}},
	}
}
