package realtimevoice

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// Defaults for the WebSocket plumbing.
const (
	defaultWriteTimeout = 10 * time.Second
	defaultPongWait     = 5 * time.Minute
	defaultPingInterval = 30 * time.Second
	defaultReadLimit    = 1 << 20 // 1 MiB
)

// Server answers BidiGenerateContent WebSocket connections. Register it on the
// Gemini Live paths:
//
//	/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent
//	/ws/google.ai.generativelanguage.v1beta1.GenerativeService/BidiGenerateContent
//
// A client that produced a working session against Gemini should need no code
// change; only the host, the key and the model path differ.
type Server struct {
	// Engine builds the backend for one session. It receives the client's setup
	// frame and may return an error to reject the session before setupComplete.
	Engine func(ctx context.Context, setup *Setup) (Engine, error)
	// APIKey, when non-empty, must match the ?key= query parameter or the
	// Authorization: Bearer token. Empty accepts any caller (dev mode).
	APIKey string
	// Logger defaults to slog.Default().
	Logger *slog.Logger

	WriteTimeout time.Duration
	PongWait     time.Duration
	PingInterval time.Duration
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		http.Error(w, "invalid API key", http.StatusUnauthorized)
		return
	}
	if s.Engine == nil {
		http.Error(w, "server is not configured with an engine", http.StatusInternalServerError)
		return
	}
	up := websocket.Upgrader{
		ReadBufferSize:  4096,
		WriteBufferSize: 4096,
		// The mock is a development server; Gemini's own endpoint does not gate
		// on browser origin either.
		CheckOrigin: func(*http.Request) bool { return true },
	}
	conn, err := up.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	s.newSession(conn).run()
}

func (s *Server) authorized(r *http.Request) bool {
	if s.APIKey == "" {
		return true
	}
	if key := r.URL.Query().Get("key"); key != "" {
		return key == s.APIKey
	}
	auth := r.Header.Get("Authorization")
	if tok, ok := strings.CutPrefix(auth, "Bearer "); ok {
		return tok == s.APIKey
	}
	return false
}

func (s *Server) logger() *slog.Logger {
	if s.Logger != nil {
		return s.Logger
	}
	return slog.Default()
}

func (s *Server) writeTimeout() time.Duration {
	if s.WriteTimeout > 0 {
		return s.WriteTimeout
	}
	return defaultWriteTimeout
}

func (s *Server) pongWait() time.Duration {
	if s.PongWait > 0 {
		return s.PongWait
	}
	return defaultPongWait
}

func (s *Server) pingInterval() time.Duration {
	if s.PingInterval > 0 {
		return s.PingInterval
	}
	return defaultPingInterval
}

func (s *Server) newSession(conn *websocket.Conn) *session {
	ctx, cancel := context.WithCancel(context.Background())
	return &session{
		srv:    s,
		conn:   conn,
		log:    s.logger(),
		ctx:    ctx,
		cancel: cancel,
	}
}

// session is one WebSocket. The reader goroutine owns the socket lifecycle;
// each assistant turn runs on its own goroutine so a barge-in frame can cancel
// it without waiting for the reader.
type session struct {
	srv    *Server
	conn   *websocket.Conn
	log    *slog.Logger
	ctx    context.Context
	cancel context.CancelFunc

	writeMu sync.Mutex

	mu     sync.Mutex
	engine Engine
	setup  *Setup
	turn   *activeTurn
	turnID uint64
}

type activeTurn struct {
	id     uint64
	cancel context.CancelFunc
}

func (s *session) run() {
	defer s.close()

	conn := s.conn
	conn.SetReadLimit(defaultReadLimit)
	_ = conn.SetReadDeadline(time.Now().Add(s.srv.pongWait()))
	conn.SetPongHandler(func(string) error {
		return conn.SetReadDeadline(time.Now().Add(s.srv.pongWait()))
	})

	stopPing := make(chan struct{})
	defer close(stopPing)
	go s.pingLoop(stopPing)

	for {
		msgType, data, err := conn.ReadMessage()
		if err != nil {
			return
		}
		if msgType != websocket.TextMessage {
			s.closeWith(websocket.CloseUnsupportedData, "binary frames are not part of the Live protocol")
			return
		}
		_ = conn.SetReadDeadline(time.Now().Add(s.srv.pongWait()))

		var ev ClientEvent
		if err := json.Unmarshal(data, &ev); err != nil {
			s.closeWith(websocket.CloseUnsupportedData, "invalid JSON frame")
			return
		}
		if !s.dispatch(&ev) {
			return
		}
	}
}

func (s *session) pingLoop(stop <-chan struct{}) {
	t := time.NewTicker(s.srv.pingInterval())
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
			if err := s.conn.WriteControl(websocket.PingMessage, nil,
				time.Now().Add(s.srv.writeTimeout())); err != nil {
				return
			}
		}
	}
}

// dispatch handles one frame and reports whether the session should continue.
func (s *session) dispatch(ev *ClientEvent) bool {
	switch {
	case ev.Setup != nil:
		return s.handleSetup(ev.Setup)
	case ev.RealtimeInput != nil:
		return s.handleRealtimeInput(ev.RealtimeInput)
	case ev.ClientContent != nil:
		return s.handleClientContent(ev.ClientContent)
	case ev.ToolResponse != nil:
		return s.handleToolResponse(ev.ToolResponse)
	case ev.ContextUpdate != nil:
		return s.handleContextUpdate(ev.ContextUpdate)
	default:
		// An empty frame is not part of the protocol; ignoring it keeps the
		// session alive where closing would be harsher than Gemini.
		return true
	}
}

func (s *session) handleSetup(setup *Setup) bool {
	s.mu.Lock()
	already := s.engine != nil
	s.mu.Unlock()
	if already {
		s.closeWith(websocket.ClosePolicyViolation, "duplicate setup")
		return false
	}
	if strings.TrimSpace(setup.Model) == "" {
		s.closeWith(websocket.ClosePolicyViolation, "setup.model is required")
		return false
	}

	engine, err := s.srv.Engine(s.ctx, setup)
	if err != nil {
		s.log.Warn("engine rejected session", "err", err, "model", setup.Model)
		s.closeWith(websocket.CloseInternalServerErr, "engine unavailable")
		return false
	}

	s.mu.Lock()
	s.engine = engine
	s.setup = setup
	s.mu.Unlock()

	if err := s.send(ServerEvent{SetupComplete: &struct{}{}}); err != nil {
		return false
	}
	if setup.SessionResumption != nil {
		_ = s.send(ServerEvent{SessionResumptionUpdate: &SessionResumptionUpdate{
			NewHandle: newHandle(),
			Resumable: true,
		}})
	}
	return true
}

func (s *session) handleRealtimeInput(in *RealtimeInput) bool {
	if !s.ready() {
		return false
	}
	engine := s.engineOrNil()
	if engine == nil {
		return true
	}

	if in.ActivityStart != nil {
		s.interrupt()
	}

	chunks := in.MediaChunks
	if in.Audio != nil {
		chunks = append(chunks, *in.Audio)
	}
	for _, b := range chunks {
		if b.MimeType != "" && !strings.HasPrefix(b.MimeType, "audio/pcm") {
			s.closeWith(websocket.CloseUnsupportedData, "only audio/pcm input is supported")
			return false
		}
		pcm, err := b.PCM()
		if err != nil {
			s.closeWith(websocket.CloseUnsupportedData, "invalid base64 audio")
			return false
		}
		if err := engine.PushAudio(s.ctx, pcm, &turnSink{s: s, id: 0}); err != nil {
			s.log.Warn("push audio", "err", err)
			s.closeWith(websocket.CloseInternalServerErr, "audio pipeline failed")
			return false
		}
	}

	if in.Text != "" {
		s.startTurn(func(ctx context.Context, sink Sink) error {
			return engine.Respond(ctx, in.Text, sink)
		})
	}

	if in.ActivityEnd != nil || in.AudioStreamEnd {
		s.commitUserTurn()
	}
	return true
}

func (s *session) commitUserTurn() {
	engine := s.engineOrNil()
	if engine == nil {
		return
	}
	text, err := engine.CommitUserTurn(s.ctx, &turnSink{s: s, id: 0})
	if err != nil {
		s.log.Warn("commit user turn", "err", err)
		return
	}
	if strings.TrimSpace(text) == "" {
		return
	}
	s.startTurn(func(ctx context.Context, sink Sink) error {
		return engine.Respond(ctx, text, sink)
	})
}

func (s *session) handleClientContent(cc *ClientContent) bool {
	if !s.ready() {
		return false
	}
	engine := s.engineOrNil()
	if engine == nil {
		return true
	}
	text := contentText(cc.Turns)
	if text == "" {
		return true
	}
	if cc.TurnComplete {
		s.startTurn(func(ctx context.Context, sink Sink) error {
			return engine.Respond(ctx, text, sink)
		})
	}
	return true
}

func (s *session) handleToolResponse(tr *ToolResponse) bool {
	if !s.ready() {
		return false
	}
	engine := s.engineOrNil()
	if engine == nil || len(tr.FunctionResponses) == 0 {
		return true
	}
	s.startTurn(func(ctx context.Context, sink Sink) error {
		return engine.SubmitToolResults(ctx, tr.FunctionResponses, sink)
	})
	return true
}

func (s *session) handleContextUpdate(cu *ContextUpdate) bool {
	if !s.ready() {
		return false
	}
	engine := s.engineOrNil()
	if engine == nil || cu.Tools == nil {
		return true
	}
	if err := engine.UpdateTools(s.ctx, cu.Tools.Tools); err != nil {
		if errors.Is(err, ErrToolUpdateUnsupported) {
			s.log.Warn("tool update unsupported; session keeps its original tools")
			return true
		}
		s.log.Warn("tool update", "err", err)
		s.closeWith(websocket.CloseInternalServerErr, "tool update failed")
		return false
	}
	return true
}

// ready reports whether setup has completed. A frame before setup closes the
// session, matching the Live API's requirement that setup come first.
func (s *session) ready() bool {
	s.mu.Lock()
	ok := s.engine != nil
	s.mu.Unlock()
	if !ok {
		s.closeWith(websocket.ClosePolicyViolation, "setup must be the first frame")
	}
	return ok
}

func (s *session) engineOrNil() Engine {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.engine
}

// startTurn runs one assistant turn on its own goroutine. Starting a turn while
// another is still speaking is a barge-in: the old turn is cancelled and the
// client is told to stop playback.
func (s *session) startTurn(run func(context.Context, Sink) error) {
	s.interrupt()

	ctx, cancel := context.WithCancel(s.ctx)
	s.mu.Lock()
	s.turnID++
	id := s.turnID
	s.turn = &activeTurn{id: id, cancel: cancel}
	s.mu.Unlock()

	go func() {
		err := run(ctx, &turnSink{s: s, id: id})
		cancel()
		if err != nil && ctx.Err() == nil && !errors.Is(err, context.Canceled) {
			s.log.Warn("assistant turn failed", "err", err)
		}
		s.clearTurn(id)
	}()
}

// interrupt cancels the active turn, if any, and tells the client to stop
// playing what it already received.
func (s *session) interrupt() {
	s.mu.Lock()
	turn := s.turn
	s.turn = nil
	s.mu.Unlock()
	if turn == nil {
		return
	}
	turn.cancel()
	_ = s.send(ServerEvent{ServerContent: &ServerContent{Interrupted: true}})
}

func (s *session) clearTurn(id uint64) {
	s.mu.Lock()
	if s.turn != nil && s.turn.id == id {
		s.turn = nil
	}
	s.mu.Unlock()
}

func (s *session) send(ev ServerEvent) error {
	data, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	_ = s.conn.SetWriteDeadline(time.Now().Add(s.srv.writeTimeout()))
	return s.conn.WriteMessage(websocket.TextMessage, data)
}

func (s *session) closeWith(code int, reason string) {
	s.writeMu.Lock()
	_ = s.conn.WriteControl(websocket.CloseMessage,
		websocket.FormatCloseMessage(code, reason), time.Now().Add(time.Second))
	s.writeMu.Unlock()
	_ = s.conn.Close()
}

func (s *session) close() {
	s.interrupt()
	s.cancel()
	s.mu.Lock()
	engine := s.engine
	s.mu.Unlock()
	if engine != nil {
		_ = engine.Close()
	}
	_ = s.conn.Close()
}

// turnSink serializes one turn's output and refuses to write once the turn has
// been cancelled (a late frame after "interrupted" would restart playback).
type turnSink struct {
	s  *session
	id uint64
}

func (t *turnSink) alive(ctx context.Context) error { return ctx.Err() }

func (t *turnSink) UserTranscription(ctx context.Context, text string, final bool) error {
	if err := t.alive(ctx); err != nil {
		return err
	}
	// The user's transcription is session-level, not playback, so it does not
	// depend on turn identity.
	return t.s.send(ServerEvent{ServerContent: &ServerContent{
		InputTranscription: &Transcription{Text: text},
		// final is carried by the client's own utterance boundary; Gemini has no
		// "final" bit on the frame, so the last fragment simply completes it.
	}})
}

func (t *turnSink) AssistantTranscription(ctx context.Context, text string) error {
	if err := t.alive(ctx); err != nil {
		return err
	}
	return t.s.send(ServerEvent{ServerContent: &ServerContent{
		OutputTranscription: &Transcription{Text: text},
	}})
}

func (t *turnSink) Audio(ctx context.Context, pcm []byte) error {
	if err := t.alive(ctx); err != nil {
		return err
	}
	return t.s.send(ServerEvent{ServerContent: &ServerContent{
		ModelTurn: NewAudioContent(pcm),
	}})
}

func (t *turnSink) ToolCall(ctx context.Context, calls []FunctionCall) error {
	if err := t.alive(ctx); err != nil {
		return err
	}
	if err := t.s.send(ServerEvent{ToolCall: &ServerToolCall{FunctionCalls: calls}}); err != nil {
		return err
	}
	// The model is not speaking while it waits for function results, so the
	// turn is no longer interruptible.
	if t.id != 0 {
		t.s.clearTurn(t.id)
	}
	return nil
}

func (t *turnSink) Usage(ctx context.Context, usage UsageMetadata) error {
	if err := t.alive(ctx); err != nil {
		return err
	}
	return t.s.send(ServerEvent{UsageMetadata: &usage})
}

func (t *turnSink) TurnComplete(ctx context.Context) error {
	if err := t.alive(ctx); err != nil {
		return err
	}
	if err := t.s.send(ServerEvent{ServerContent: &ServerContent{TurnComplete: true}}); err != nil {
		return err
	}
	if t.id != 0 {
		t.s.clearTurn(t.id)
	}
	return nil
}

// contentText concatenates the text parts of the last turn.
func contentText(turns []Content) string {
	var b strings.Builder
	for _, turn := range turns {
		for _, p := range turn.Parts {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

func newHandle() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return ""
	}
	return base64.RawURLEncoding.EncodeToString(b[:])
}
