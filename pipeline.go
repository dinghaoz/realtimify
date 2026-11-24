package realtimify

import (
	"context"
	"errors"
	"strings"
	"sync"
	"unicode/utf8"
)

// Pipeline is the real backend behind the protocol: streaming ASR for the user
// side, a text LLM for the reply, streaming TTS for the voice.
//
// It is deliberately provider-agnostic — the three stages are interfaces — so a
// vendor can be swapped without touching the protocol server.
type Pipeline struct {
	cfg Config

	mu      sync.Mutex
	history []LLMMessage
	tools   []Tool
	// toolsSet records that UpdateTools replaced the configured Tools.
	toolsSet bool
	pending  []FunctionCall // tool calls awaiting client results

	asrMu sync.Mutex
	asr   ASRSessionHandle
	// spoken is the transcript already reported to the client, so running
	// transcripts are turned into deltas.
	spoken string
}

// NewPipeline builds a pipeline from its configuration.
func NewPipeline(cfg Config) *Pipeline {
	if cfg.MaxClauseRunes <= 0 {
		cfg.MaxClauseRunes = 40
	}
	return &Pipeline{cfg: cfg}
}

// PushAudio starts a recognition stream on the first frame of an utterance and
// forwards microphone audio into it. Partials are reported as transcript
// deltas, matching the Live API's input transcription fragments.
func (p *Pipeline) PushAudio(ctx context.Context, pcm16k []byte, sink Sink) error {
	if len(pcm16k) == 0 {
		return nil
	}
	p.asrMu.Lock()
	if p.asr == nil {
		if p.cfg.ASR == nil {
			p.asrMu.Unlock()
			return errors.New("realtimify: pipeline has no ASR configured")
		}
		session, err := p.cfg.ASR.Open(ctx, p.cfg.ASROptions, func(text string) error {
			return p.reportTranscript(ctx, sink, text)
		})
		if err != nil {
			p.asrMu.Unlock()
			return err
		}
		p.asr = session
		p.spoken = ""
	}
	session := p.asr
	p.asrMu.Unlock()
	return session.Push(ctx, pcm16k)
}

// CommitUserTurn ends the utterance and returns its final transcript.
func (p *Pipeline) CommitUserTurn(ctx context.Context, sink Sink) (string, error) {
	p.asrMu.Lock()
	session := p.asr
	p.asr = nil
	p.asrMu.Unlock()
	if session == nil {
		return "", nil
	}
	defer func() { _ = session.Close() }()

	text, err := session.Finish(ctx)
	text = strings.TrimSpace(text)
	if err != nil && text == "" {
		return "", err
	}
	if text != "" {
		if err := p.reportTranscript(ctx, sink, text); err != nil {
			return "", err
		}
	}
	p.asrMu.Lock()
	p.spoken = ""
	p.asrMu.Unlock()
	return text, nil
}

// reportTranscript sends the part of a running transcript the client has not
// seen. Recognition text can be rewritten in the middle, not just extended; in
// that case the whole line is resent and the client's own accumulator wins.
func (p *Pipeline) reportTranscript(ctx context.Context, sink Sink, text string) error {
	p.asrMu.Lock()
	previous := p.spoken
	if strings.HasPrefix(text, previous) {
		delta := text[len(previous):]
		p.spoken = text
		p.asrMu.Unlock()
		if delta == "" {
			return nil
		}
		return sink.UserTranscription(ctx, delta, false)
	}
	p.spoken = text
	p.asrMu.Unlock()
	return sink.UserTranscription(ctx, text, false)
}

// Respond produces one assistant turn: text LLM → sentence split → TTS.
// Audio is emitted clause by clause, so the first sound does not wait for the
// whole reply.
func (p *Pipeline) Respond(ctx context.Context, userText string, sink Sink) error {
	p.mu.Lock()
	p.history = append(p.history, LLMMessage{Role: "user", Text: userText})
	history := append([]LLMMessage(nil), p.history...)
	tools := p.effectiveToolsLocked()
	p.mu.Unlock()

	return p.generate(ctx, LLMRequest{
		Model:             p.cfg.LLMOptions.Model,
		SystemInstruction: p.cfg.LLMOptions.SystemInstruction,
		History:           history[:len(history)-1],
		Prompt:            userText,
		Tools:             tools,
		Temperature:       p.cfg.LLMOptions.Temperature,
		MaxOutputTokens:   p.cfg.LLMOptions.MaxOutputTokens,
	}, sink)
}

// SubmitToolResults resumes the turn after the client ran its functions.
func (p *Pipeline) SubmitToolResults(ctx context.Context, results []FunctionResponse, sink Sink) error {
	p.mu.Lock()
	p.history = append(p.history, LLMMessage{Role: "user", ToolResults: results})
	history := append([]LLMMessage(nil), p.history...)
	tools := p.effectiveToolsLocked()
	p.pending = nil
	p.mu.Unlock()

	// The function results are the last history entry; there is no new prompt.
	return p.generate(ctx, LLMRequest{
		Model:             p.cfg.LLMOptions.Model,
		SystemInstruction: p.cfg.LLMOptions.SystemInstruction,
		History:           history,
		Tools:             tools,
		Temperature:       p.cfg.LLMOptions.Temperature,
		MaxOutputTokens:   p.cfg.LLMOptions.MaxOutputTokens,
	}, sink)
}

// UpdateTools swaps the function set for later turns.
func (p *Pipeline) UpdateTools(_ context.Context, tools []Tool) error {
	p.mu.Lock()
	p.tools = tools
	p.toolsSet = true
	p.mu.Unlock()
	return nil
}

// effectiveToolsLocked returns the tool set for the next request. A mid-session
// UpdateTools wins over the construction-time Tools field.
func (p *Pipeline) effectiveToolsLocked() []Tool {
	if p.toolsSet {
		return append([]Tool(nil), p.tools...)
	}
	return append([]Tool(nil), p.cfg.Tools...)
}

// Close releases the open recognition stream, if any.
func (p *Pipeline) Close() error {
	p.asrMu.Lock()
	session := p.asr
	p.asr = nil
	p.asrMu.Unlock()
	if session != nil {
		return session.Close()
	}
	return nil
}

func (p *Pipeline) generate(ctx context.Context, req LLMRequest, sink Sink) error {
	if p.cfg.LLM == nil {
		return errors.New("realtimify: pipeline has no LLM configured")
	}
	var buffered sentenceBuffer

	speak := func(clause string) error {
		clause = strings.TrimSpace(clause)
		if clause == "" {
			return nil
		}
		if err := sink.AssistantTranscription(ctx, clause); err != nil {
			return err
		}
		if p.cfg.TTS == nil {
			return nil
		}
		return p.cfg.TTS.Speak(ctx, clause, p.cfg.TTSOptions, func(pcm []byte) error {
			return sink.Audio(ctx, pcm)
		})
	}

	res, err := p.cfg.LLM.Stream(ctx, req, func(fragment string) error {
		for _, clause := range buffered.push(fragment, p.cfg.MaxClauseRunes) {
			if err := speak(clause); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	if rest := buffered.flush(); rest != "" {
		if err := speak(rest); err != nil {
			return err
		}
	}

	// Record the assistant's words as history. With tool calls the text is
	// usually empty and the calls themselves are the model turn.
	p.mu.Lock()
	p.history = append(p.history, LLMMessage{
		Role:      "assistant",
		Text:      res.Text,
		ToolCalls: res.ToolCalls,
	})
	p.pending = res.ToolCalls
	p.mu.Unlock()

	if len(res.ToolCalls) > 0 {
		return sink.ToolCall(ctx, res.ToolCalls)
	}
	if err := sink.Usage(ctx, res.Usage); err != nil {
		return err
	}
	return sink.TurnComplete(ctx)
}

// sentenceBuffer turns a token stream into clauses that are worth a TTS call.
// Punctuation ends a clause; a long run without punctuation is cut at the last
// word boundary so speech starts early.
type sentenceBuffer struct {
	b strings.Builder
}

var hardStops = map[rune]bool{
	'。': true, '！': true, '？': true, '；': true, '…': true,
	'.': true, '!': true, '?': true, ';': true, '\n': true,
}

var softStops = map[rune]bool{'，': true, ',': true, '、': true, '：': true, ':': true}

// push appends a fragment and returns every clause completed by it.
func (b *sentenceBuffer) push(fragment string, maxRunes int) []string {
	b.b.WriteString(fragment)
	var out []string
	for {
		text := b.b.String()
		cut := clauseEnd(text, maxRunes)
		if cut <= 0 {
			return out
		}
		out = append(out, strings.TrimSpace(text[:cut]))
		b.b.Reset()
		b.b.WriteString(text[cut:])
	}
}

// flush returns whatever is left.
func (b *sentenceBuffer) flush() string {
	rest := strings.TrimSpace(b.b.String())
	b.b.Reset()
	return rest
}

// clauseEnd returns the byte offset just past the clause boundary, or 0 when
// the buffer has no complete clause yet.
func clauseEnd(text string, maxRunes int) int {
	runes := 0
	lastSoft := 0
	for i, r := range text {
		runes++
		end := i + utf8.RuneLen(r)
		if hardStops[r] {
			return end
		}
		if softStops[r] {
			if runes >= 8 {
				return end
			}
			lastSoft = end
		}
		if runes >= maxRunes {
			if lastSoft > 0 {
				return lastSoft
			}
			return end
		}
	}
	return 0
}
