package realtimevoice

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// BytePlus Seed TTS 2.0 over the unidirectional HTTP endpoint. The response is
// a stream of JSON documents, each carrying a base64 audio chunk; asking for
// PCM 24 kHz means the bytes go straight to the client without decoding.
//
// Synthesis is per sentence rather than per token: the vendor bills characters
// and has no incremental-text endpoint on the international host, so the
// pipeline splits the LLM stream at punctuation and speaks each clause as it is
// complete.
const (
	bytePlusTTSDefaultURL = "https://voice.ap-southeast-1.bytepluses.com/api/v3/tts/unidirectional"
	bytePlusTTSEndpoint   = "seed-tts-2.0"
	defaultBytePlusVoice  = "zh_female_vv_uranus_bigtts"
	// bytePlusTTSFinishCode terminates the stream: the vendor sends it as a
	// final document, after the audio, with a non-zero code.
	bytePlusTTSFinishCode = 20000000
)

// BytePlusTTS synthesizes text into 24 kHz PCM.
type BytePlusTTS struct {
	APIKey     string
	URL        string
	ResourceID string
	// Speaker is the voice type, e.g. "zh_female_vv_uranus_bigtts".
	Speaker string
	// Language is an optional hint such as "zh" or "en".
	Language string
	// Model is an optional TTS model name; empty uses the account default.
	Model string
	HTTP  *http.Client
}

// Speak implements TTS. Voice, language and model come from opts and fall back
// to the provider defaults, so a session can pick its 音色 without a new client.
func (t *BytePlusTTS) Speak(ctx context.Context, text string, opts TTSOptions, onPCM func(pcm []byte) error) error {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	if t.APIKey == "" {
		return fmt.Errorf("realtimevoice: byteplus tts api key is empty")
	}
	endpoint := t.URL
	if endpoint == "" {
		endpoint = bytePlusTTSDefaultURL
	}
	resource := t.ResourceID
	if resource == "" {
		resource = bytePlusTTSEndpoint
	}
	speaker := opts.Voice
	if speaker == "" {
		speaker = t.Speaker
	}
	if speaker == "" {
		speaker = defaultBytePlusVoice
	}
	sampleRate := opts.SampleRate
	if sampleRate <= 0 {
		sampleRate = OutputSampleRate
	}

	params := map[string]any{
		"text":    text,
		"speaker": speaker,
		"audio_params": map[string]any{
			"format":      "pcm",
			"sample_rate": sampleRate,
		},
	}
	language := opts.Language
	if language == "" {
		language = t.Language
	}
	if language != "" {
		params["language"] = language
	}
	model := opts.Model
	if model == "" {
		model = t.Model
	}
	if model != "" {
		params["model"] = model
	}
	body, err := json.Marshal(map[string]any{"req_params": params})
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	// x-api-key, never the URL: this key is shared with ASR and must not land in
	// access logs.
	req.Header.Set("x-api-key", t.APIKey)
	req.Header.Set("X-Api-Resource-Id", resource)

	client := t.HTTP
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Minute}
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("realtimevoice: byteplus tts returned %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
	}
	return readBytePlusTTS(resp.Body, onPCM)
}

// readBytePlusTTS decodes the concatenated JSON documents of a synthesis
// response. A non-zero code is an error even when audio already played: the
// clause would otherwise be silently truncated.
func readBytePlusTTS(body io.Reader, onPCM func([]byte) error) error {
	dec := json.NewDecoder(body)
	for {
		var chunk struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
			Data    string `json:"data"`
		}
		if err := dec.Decode(&chunk); err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("realtimevoice: decode byteplus tts: %w", err)
		}
		switch chunk.Code {
		case bytePlusTTSFinishCode:
			return nil
		case 0:
			// audio below
		default:
			return fmt.Errorf("realtimevoice: byteplus tts code %d: %s", chunk.Code, chunk.Message)
		}
		if chunk.Data == "" {
			continue
		}
		pcm, err := base64.StdEncoding.DecodeString(chunk.Data)
		if err != nil {
			return fmt.Errorf("realtimevoice: decode byteplus tts audio: %w", err)
		}
		if len(pcm) == 0 {
			continue
		}
		if onPCM != nil {
			if err := onPCM(pcm); err != nil {
				return err
			}
		}
	}
}
