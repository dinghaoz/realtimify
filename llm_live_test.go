//go:build livellm

package realtimevoice

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// liveLLM builds a client from the environment. The Express-mode key works on
// the global Vertex host, which is why Location defaults to "global".
//
//	VERTEX_API_KEY=... VERTEX_PROJECT_ID=... go test -tags livellm ./internal/realtimevoice -run Live -v
func liveLLM(t *testing.T) *GeminiLLM {
	t.Helper()
	key := os.Getenv("VERTEX_API_KEY")
	project := os.Getenv("VERTEX_PROJECT_ID")
	if key == "" || project == "" {
		t.Skip("VERTEX_API_KEY / VERTEX_PROJECT_ID not set; skipping live Gemini test")
	}
	model := os.Getenv("VERTEX_LLM_MODEL")
	if model == "" {
		model = "gemini-2.5-flash"
	}
	budget := 0
	return &GeminiLLM{
		Project:        project,
		Location:       "global",
		Model:          model,
		APIKey:         key,
		ThinkingBudget: &budget,
	}
}

// TestGeminiLLMLiveText proves the request shape the mock pipeline sends is
// accepted by the real endpoint and that streaming fragments, usage and
// finishReason come back in the shape readGeminiSSE expects.
func TestGeminiLLMLiveText(t *testing.T) {
	llm := liveLLM(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var fragments []string
	res, err := llm.Stream(ctx, LLMRequest{
		SystemInstruction: "You are a voice assistant. Answer in one short sentence.",
		History: []LLMMessage{
			{Role: "user", Text: "你好"},
			{Role: "assistant", Text: "你好，有什么可以帮你？"},
		},
		Prompt:          "说一句八个字以内的话",
		Temperature:     0.2,
		MaxOutputTokens: 64,
	}, func(s string) error {
		fragments = append(fragments, s)
		return nil
	})
	require.NoError(t, err)

	t.Logf("fragments=%d text=%q finish=%s usage=%+v", len(fragments), res.Text, res.FinishReason, res.Usage)
	require.NotEmpty(t, strings.TrimSpace(res.Text))
	require.Equal(t, res.Text, strings.Join(fragments, ""), "streamed fragments must reconstruct the reply")
	require.Greater(t, res.Usage.PromptTokenCount, 0)
	require.Greater(t, res.Usage.CandidatesTokenCount, 0)
	require.Empty(t, res.ToolCalls)
}

// TestGeminiLLMLiveToolCall proves a declared function is returned as a
// functionCall part rather than as text.
func TestGeminiLLMLiveToolCall(t *testing.T) {
	llm := liveLLM(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	res, err := llm.Stream(ctx, LLMRequest{
		SystemInstruction: "You control a meeting room. Always use the tool when asked to book.",
		Prompt:            "帮我订明天下午三点的会议室，人数 6。",
		Tools: []Tool{{FunctionDeclarations: []FunctionDeclaration{{
			Name:        "book_room",
			Description: "Book a meeting room.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"hour":{"type":"integer"},"people":{"type":"integer"}},"required":["hour","people"]}`),
		}}}},
		Temperature: 0,
	}, nil)
	require.NoError(t, err)

	t.Logf("text=%q toolCalls=%+v usage=%+v", res.Text, res.ToolCalls, res.Usage)
	require.Len(t, res.ToolCalls, 1)
	require.Equal(t, "book_room", res.ToolCalls[0].Name)
	require.NotEmpty(t, res.ToolCalls[0].Args)
}

// TestPipelineLiveToolTurn drives one turn through the real pipeline so a
// provider-side rejection shows up as an error instead of a silent stall.
func TestPipelineLiveToolTurn(t *testing.T) {
	llm := liveLLM(t)
	p := NewPipeline(Config{
		LLM:        llm,
		LLMOptions: LLMOptions{SystemInstruction: "你是会议室助手。订会议室必须调用 book_room。"},
		Tools: []Tool{{FunctionDeclarations: []FunctionDeclaration{{
			Name:        "book_room",
			Description: "Book a meeting room.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"hour":{"type":"integer"},"people":{"type":"integer"}},"required":["hour","people"]}`),
		}}}},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	sink := &recordingSink{}
	err := p.Respond(ctx, "帮我订今天下午三点的会议室，六个人", sink)
	require.NoError(t, err)
	require.Len(t, sink.toolCalls, 1)
	require.Equal(t, "book_room", sink.toolCalls[0].Name)
}
