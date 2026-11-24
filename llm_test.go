package geminilive

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGeminiLLMStreamsText(t *testing.T) {
	var gotPath, gotAlt, gotKey string
	var gotBody geminiGenerateRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAlt = r.URL.Query().Get("alt")
		gotKey = r.Header.Get("x-goog-api-key")
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"你好\"}]}}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"呀\"}]},\"finishReason\":\"STOP\"}],"+
			"\"usageMetadata\":{\"promptTokenCount\":12,\"candidatesTokenCount\":2,\"thoughtsTokenCount\":5,\"totalTokenCount\":19}}\n\n")
	}))
	t.Cleanup(srv.Close)

	budget := 0
	llm := &GeminiLLM{
		BaseURL: srv.URL, Project: "proj", Location: "global", Model: "gemini-x",
		APIKey: "test-key", ThinkingBudget: &budget,
	}

	var fragments []string
	res, err := llm.Stream(context.Background(), LLMRequest{
		SystemInstruction: "be brief",
		History: []LLMMessage{
			{Role: "user", Text: "第一句"},
			{Role: "assistant", Text: "第一答"},
		},
		Prompt: "第二句",
		Tools:  []Tool{{FunctionDeclarations: []FunctionDeclaration{{Name: "book"}}}},
	}, func(s string) error {
		fragments = append(fragments, s)
		return nil
	})
	require.NoError(t, err)

	require.Equal(t, "/v1/projects/proj/locations/global/publishers/google/models/gemini-x:streamGenerateContent", gotPath)
	require.Equal(t, "sse", gotAlt)
	require.Equal(t, "test-key", gotKey, "key must travel in a header, not the URL")
	require.NotContains(t, gotAlt, "test-key")

	require.Equal(t, "be brief", gotBody.SystemInstruction.Parts[0].Text)
	require.Len(t, gotBody.Contents, 3)
	require.Equal(t, "user", gotBody.Contents[0].Role)
	require.Equal(t, "model", gotBody.Contents[1].Role, "assistant history maps to the model role")
	require.Equal(t, "第二句", gotBody.Contents[2].Parts[0].Text)
	require.Equal(t, "book", gotBody.Tools[0].FunctionDeclarations[0].Name)
	require.NotNil(t, gotBody.GenerationConfig.ThinkingConfig, "thinking must be disabled explicitly")
	require.Equal(t, 0, *gotBody.GenerationConfig.ThinkingConfig.ThinkingBudget)

	require.Equal(t, []string{"你好", "呀"}, fragments)
	require.Equal(t, "你好呀", res.Text)
	require.Equal(t, "STOP", res.FinishReason)
	require.Equal(t, 12, res.Usage.PromptTokenCount)
	require.Equal(t, 7, res.Usage.CandidatesTokenCount, "thoughts are billed as output")
	require.Equal(t, 19, res.Usage.TotalTokenCount)
}

// TestGeminiLLMRequestModelOverridesDefault pins the per-request model: a turn
// may pick its own model, and the provider field is only the fallback.
func TestGeminiLLMRequestModelOverridesDefault(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]}}]}\n\n")
	}))
	t.Cleanup(srv.Close)

	llm := &GeminiLLM{BaseURL: srv.URL, Project: "proj", Model: "provider-default", APIKey: "k"}
	_, err := llm.Stream(context.Background(), LLMRequest{Prompt: "hi", Model: "per-request"}, nil)
	require.NoError(t, err)
	require.Contains(t, gotPath, "/models/per-request:")

	_, err = llm.Stream(context.Background(), LLMRequest{Prompt: "hi"}, nil)
	require.NoError(t, err)
	require.Contains(t, gotPath, "/models/provider-default:")
}

func TestGeminiLLMParsesToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"candidates":[{"content":{"role":"model","parts":[`+
			`{"functionCall":{"name":"book","args":{"hour":15}}},`+
			`{"text":"ignored because a tool call ends the turn"}]}}]}`+"\n\n")
	}))
	t.Cleanup(srv.Close)

	llm := &GeminiLLM{BaseURL: srv.URL, Project: "p", Model: "m", APIKey: "k"}
	res, err := llm.Stream(context.Background(), LLMRequest{Prompt: "book it"}, nil)
	require.NoError(t, err)

	require.Len(t, res.ToolCalls, 1)
	require.Equal(t, "book", res.ToolCalls[0].Name)
	require.Equal(t, "call_1", res.ToolCalls[0].ID)
	require.JSONEq(t, `{"hour":15}`, string(res.ToolCalls[0].Args))
}

func TestGeminiLLMSkipsThoughtParts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"candidates":[{"content":{"parts":[`+
			`{"text":"hidden reasoning","thought":true},{"text":"visible"}]}}]}`+"\n\n")
	}))
	t.Cleanup(srv.Close)

	llm := &GeminiLLM{BaseURL: srv.URL, Project: "p", Model: "m", APIKey: "k"}
	var seen []string
	res, err := llm.Stream(context.Background(), LLMRequest{Prompt: "x"}, func(s string) error {
		seen = append(seen, s)
		return nil
	})
	require.NoError(t, err)
	require.Equal(t, []string{"visible"}, seen)
	require.Equal(t, "visible", res.Text)
}

func TestGeminiLLMHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":{"code":403,"message":"denied"}}`, http.StatusForbidden)
	}))
	t.Cleanup(srv.Close)

	llm := &GeminiLLM{BaseURL: srv.URL, Project: "p", Model: "m", APIKey: "secret-key"}
	_, err := llm.Stream(context.Background(), LLMRequest{Prompt: "x"}, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "403")
	require.NotContains(t, err.Error(), "secret-key")
}

func TestGeminiLLMRequiresConfig(t *testing.T) {
	_, err := (&GeminiLLM{Project: "p", Model: "m"}).Stream(context.Background(), LLMRequest{Prompt: "x"}, nil)
	require.Error(t, err)

	_, err = (&GeminiLLM{APIKey: "k", Model: "m"}).Stream(context.Background(), LLMRequest{Prompt: "x"}, nil)
	require.Error(t, err)
}

func TestGeminiLLMBlockedPrompt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"promptFeedback":{"blockReason":"SAFETY"}}`+"\n\n")
	}))
	t.Cleanup(srv.Close)

	llm := &GeminiLLM{BaseURL: srv.URL, Project: "p", Model: "m", APIKey: "k"}
	_, err := llm.Stream(context.Background(), LLMRequest{Prompt: "x"}, nil)
	require.ErrorContains(t, err, "SAFETY")
}
