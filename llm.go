package realtimevoice

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// TextLLM is the text-generation stage of the pipeline. It streams the
// assistant's words so TTS can start on the first clause instead of waiting
// for the whole reply.
type TextLLM interface {
	// Stream generates one reply. onText is called for every incremental text
	// fragment, in order. The returned result repeats the full text plus any
	// tool calls and token usage.
	Stream(ctx context.Context, req LLMRequest, onText func(string) error) (LLMResult, error)
}

// LLMRequest is one turn of the conversation.
type LLMRequest struct {
	// Model selects the text model for this turn; empty uses the provider's
	// configured default.
	Model             string
	SystemInstruction string
	// History is oldest-first; it does not include the current Prompt.
	History []LLMMessage
	Prompt  string
	Tools   []Tool

	Temperature     float64
	MaxOutputTokens int
}

// LLMMessage is one prior turn. Exactly one of Text, ToolCalls and ToolResults
// is normally set; tool turns carry calls (assistant) or results (user).
type LLMMessage struct {
	Role string // "user" or "model"
	Text string
	// ToolCalls are the function calls the model made on this turn.
	ToolCalls []FunctionCall
	// ToolResults are the function responses the client returned.
	ToolResults []FunctionResponse
}

// LLMResult is the completed reply.
type LLMResult struct {
	Text         string
	ToolCalls    []FunctionCall
	Usage        UsageMetadata
	FinishReason string
}

// GeminiLLM calls Vertex AI's generateContent with an API key (an
// "Express mode" key works: send it in x-goog-api-key, not the query string).
//
// It speaks the arrive-by-global-endpoint shape:
//
//	{base}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:streamGenerateContent?alt=sse
type GeminiLLM struct {
	BaseURL  string // default https://aiplatform.googleapis.com
	Project  string
	Location string // default global
	Model    string
	APIKey   string
	HTTP     *http.Client

	// ThinkingBudget disables or caps reasoning tokens. Realtime voice wants 0:
	// default thinking costs hundreds of output tokens and adds latency before
	// the first spoken word.
	ThinkingBudget *int
}

const geminiDefaultBaseURL = "https://aiplatform.googleapis.com"

// Stream implements TextLLM.
func (g *GeminiLLM) Stream(ctx context.Context, req LLMRequest, onText func(string) error) (LLMResult, error) {
	if g.APIKey == "" {
		return LLMResult{}, errors.New("realtimevoice: gemini api key is empty")
	}
	model := req.Model
	if model == "" {
		model = g.Model
	}
	if model == "" {
		return LLMResult{}, errors.New("realtimevoice: gemini model is empty")
	}
	endpoint, err := g.endpoint(model)
	if err != nil {
		return LLMResult{}, err
	}
	body, err := json.Marshal(g.buildRequest(req))
	if err != nil {
		return LLMResult{}, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return LLMResult{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	// The key travels in a header, never the URL: query strings end up in proxy
	// and access logs.
	httpReq.Header.Set("x-goog-api-key", g.APIKey)

	client := g.HTTP
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return LLMResult{}, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return LLMResult{}, fmt.Errorf("realtimevoice: gemini returned %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
	}

	return readGeminiSSE(resp.Body, onText)
}

func (g *GeminiLLM) endpoint(model string) (string, error) {
	base := strings.TrimRight(g.BaseURL, "/")
	if base == "" {
		base = geminiDefaultBaseURL
	}
	loc := g.Location
	if loc == "" {
		loc = "global"
	}
	if g.Project == "" {
		return "", errors.New("realtimevoice: gemini project is empty")
	}
	u, err := url.Parse(fmt.Sprintf("%s/v1/projects/%s/locations/%s/publishers/google/models/%s:streamGenerateContent",
		base, url.PathEscape(g.Project), url.PathEscape(loc), url.PathEscape(model)))
	if err != nil {
		return "", err
	}
	q := u.Query()
	q.Set("alt", "sse")
	u.RawQuery = q.Encode()
	return u.String(), nil
}

func (g *GeminiLLM) buildRequest(req LLMRequest) geminiGenerateRequest {
	out := geminiGenerateRequest{
		Contents: make([]geminiContent, 0, len(req.History)+1),
	}
	if req.SystemInstruction != "" {
		out.SystemInstruction = &geminiContent{Parts: []geminiPart{{Text: req.SystemInstruction}}}
	}
	for _, m := range req.History {
		role := m.Role
		if role == "assistant" {
			role = "model"
		}
		parts := make([]geminiPart, 0, 1+len(m.ToolCalls)+len(m.ToolResults))
		if m.Text != "" {
			parts = append(parts, geminiPart{Text: m.Text})
		}
		for _, call := range m.ToolCalls {
			parts = append(parts, geminiPart{FunctionCall: &geminiFunctionCall{
				Name: call.Name,
				Args: call.Args,
			}})
		}
		for _, result := range m.ToolResults {
			parts = append(parts, geminiPart{FunctionResponse: &geminiFunctionResponse{
				Name:     result.Name,
				Response: result.Response,
			}})
		}
		if len(parts) == 0 {
			continue
		}
		out.Contents = append(out.Contents, geminiContent{Role: role, Parts: parts})
	}
	if req.Prompt != "" {
		out.Contents = append(out.Contents, geminiContent{Role: "user", Parts: []geminiPart{{Text: req.Prompt}}})
	}
	out.Tools = req.Tools

	gen := &geminiGenerationConfig{}
	if req.Temperature > 0 {
		gen.Temperature = &req.Temperature
	}
	if req.MaxOutputTokens > 0 {
		gen.MaxOutputTokens = req.MaxOutputTokens
	}
	if g.ThinkingBudget != nil {
		gen.ThinkingConfig = &geminiThinkingConfig{ThinkingBudget: g.ThinkingBudget}
	}
	if gen.Temperature != nil || gen.MaxOutputTokens > 0 || gen.ThinkingConfig != nil {
		out.GenerationConfig = gen
	}
	return out
}

// --- wire shapes (generateContent) ---

type geminiGenerateRequest struct {
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	Contents          []geminiContent         `json:"contents"`
	Tools             []Tool                  `json:"tools,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
	Thought          bool                    `json:"thought,omitempty"`
}

type geminiFunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type geminiFunctionResponse struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type geminiGenerationConfig struct {
	Temperature     *float64              `json:"temperature,omitempty"`
	MaxOutputTokens int                   `json:"maxOutputTokens,omitempty"`
	ThinkingConfig  *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
}

type geminiThinkingConfig struct {
	ThinkingBudget *int `json:"thinkingBudget,omitempty"`
}

type geminiStreamChunk struct {
	Candidates []struct {
		Content      geminiContent `json:"content"`
		FinishReason string        `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata *struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		ThoughtsTokenCount   int `json:"thoughtsTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
}

func readGeminiSSE(body io.Reader, onText func(string) error) (LLMResult, error) {
	var result LLMResult
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 4<<20)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var chunk geminiStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return result, fmt.Errorf("realtimevoice: decode gemini chunk: %w", err)
		}
		if chunk.PromptFeedback != nil && chunk.PromptFeedback.BlockReason != "" {
			return result, fmt.Errorf("realtimevoice: prompt blocked: %s", chunk.PromptFeedback.BlockReason)
		}
		for _, cand := range chunk.Candidates {
			if cand.FinishReason != "" {
				result.FinishReason = cand.FinishReason
			}
			for _, part := range cand.Content.Parts {
				if part.FunctionCall != nil {
					result.ToolCalls = append(result.ToolCalls, FunctionCall{
						ID:   fmt.Sprintf("call_%d", len(result.ToolCalls)+1),
						Name: part.FunctionCall.Name,
						Args: part.FunctionCall.Args,
					})
					continue
				}
				if part.Thought || part.Text == "" {
					continue
				}
				result.Text += part.Text
				if onText != nil {
					if err := onText(part.Text); err != nil {
						return result, err
					}
				}
			}
		}
		if u := chunk.UsageMetadata; u != nil {
			// Reasoning tokens are billed as output; fold them in so cost matches
			// the invoice.
			cand := u.CandidatesTokenCount + u.ThoughtsTokenCount
			result.Usage = UsageMetadata{
				PromptTokenCount:     u.PromptTokenCount,
				CandidatesTokenCount: cand,
				TotalTokenCount:      u.TotalTokenCount,
			}
			if u.PromptTokenCount > 0 || cand > 0 {
				result.Usage.PromptTokensDetails = []TokenCount{{Modality: "TEXT", TokenCount: u.PromptTokenCount}}
				result.Usage.CandidateTokenDetails = []TokenCount{{Modality: "TEXT", TokenCount: cand}}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return result, err
	}
	return result, nil
}
