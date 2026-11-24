package realtimevoice

import (
	"context"
	crand "crypto/rand"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// BytePlus Seed ASR 2.0 (international). Streaming WebSocket, binary SAUC
// framing. The key comes from the Seed Speech console and travels in
// x-api-key.
const (
	bytePlusASRDefaultURL      = "wss://voice.ap-southeast-1.bytepluses.com/api/v3/sauc/bigmodel_async"
	bytePlusASRDefaultResource = "volc.seedasr.sauc.duration"
)

// SAUC frame layout: one header byte, a message-type nibble plus flags, a
// serialization nibble plus compression, one reserved byte — then optional
// sequence, then a big-endian payload length and the payload.
const (
	saucVersion       = 0x11
	saucTypeFullReq   = 0x01
	saucTypeAudio     = 0x02
	saucTypeServer    = 0x09
	saucTypeError     = 0x0F
	saucFlagSequence  = 0x01
	saucFlagLast      = 0x02
	saucSerialization = 0x01 // JSON
)

// BytePlusASR opens one streaming recognition session per spoken utterance.
// Recognition is cumulative: every server frame repeats the transcript so far,
// so partials are reported as running text and the last frame is final.
type BytePlusASR struct {
	APIKey     string
	URL        string
	ResourceID string
	Dialer     *websocket.Dialer
	// Model is the ASR model name; "bigmodel" is the only documented value.
	Model string
	// Language is the default hint ("zh-CN", "en-US", ...) used when
	// ASROptions.Language is empty. Empty auto-detects.
	Language string
}

// ASRSession is one open recognition stream.
type ASRSession struct {
	conn     *websocket.Conn
	mu       sync.Mutex // serializes writes and guards finished
	finished bool

	readDone chan struct{}
	finalCh  chan string
	errMu    sync.Mutex
	err      error

	// lastText is the newest transcript the server reported. It is the best
	// available answer when the final frame does not arrive in time.
	lastTextMu sync.Mutex
	lastText   string
}

// Open implements ASR. The language hint and sample rate come from opts so one
// server can serve sessions in different languages without a new provider.
func (b *BytePlusASR) Open(ctx context.Context, opts ASROptions, onPartial func(text string) error) (ASRSessionHandle, error) {
	if b.APIKey == "" {
		return nil, errors.New("realtimevoice: byteplus asr api key is empty")
	}
	endpoint := b.URL
	if endpoint == "" {
		endpoint = bytePlusASRDefaultURL
	}
	resource := b.ResourceID
	if resource == "" {
		resource = bytePlusASRDefaultResource
	}
	dialer := b.Dialer
	if dialer == nil {
		dialer = websocket.DefaultDialer
	}
	headers := http.Header{
		"X-Api-Key":         {b.APIKey},
		"X-Api-Resource-Id": {resource},
		"X-Api-Connect-Id":  {newASRHandle()},
	}
	conn, resp, err := dialer.DialContext(ctx, endpoint, headers)
	if err != nil {
		body := ""
		if resp != nil {
			body = resp.Status
		}
		return nil, fmt.Errorf("realtimevoice: byteplus asr dial: %w (%s)", err, body)
	}

	model := b.Model
	if model == "" {
		model = "bigmodel"
	}
	sampleRate := opts.SampleRate
	if sampleRate <= 0 {
		sampleRate = InputSampleRate
	}
	req := map[string]any{
		"user":  map[string]any{"uid": "realtimevoice"},
		"audio": map[string]any{"format": "pcm", "sample_rate": sampleRate, "channel": 1, "bits": 16, "codec": "raw"},
		"request": map[string]any{
			"reqid":           newASRHandle(),
			"sequence":        1,
			"model_name":      model,
			"result_type":     "full",
			"show_utterances": true,
		},
	}
	language := opts.Language
	if language == "" {
		language = b.Language
	}
	if language != "" {
		req["request"].(map[string]any)["language"] = language
	}
	payload, err := json.Marshal(req)
	if err != nil {
		_ = conn.Close()
		return nil, err
	}
	if err := conn.WriteMessage(websocket.BinaryMessage, saucFrame(saucTypeFullReq, saucFlagSequence, saucSerialization, 1, payload)); err != nil {
		_ = conn.Close()
		return nil, fmt.Errorf("realtimevoice: byteplus asr setup: %w", err)
	}

	s := &ASRSession{
		conn:     conn,
		readDone: make(chan struct{}),
		finalCh:  make(chan string, 1),
	}
	go s.readLoop(onPartial)
	return s, nil
}

// Push sends 16 kHz mono PCM. Chunks should be 100-200 ms; the vendor rejects
// frames that are too large or too small.
func (s *ASRSession) Push(ctx context.Context, pcm16k []byte) error {
	if len(pcm16k) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.finished {
		return nil
	}
	if err := s.conn.WriteMessage(websocket.BinaryMessage, saucFrame(saucTypeAudio, 0, 0, 0, pcm16k)); err != nil {
		return fmt.Errorf("realtimevoice: byteplus asr push: %w", err)
	}
	return nil
}

// Finish closes the audio stream and waits for the final transcript. A timeout
// still returns the newest partial: a short trailing word is better than
// dropping the whole utterance.
func (s *ASRSession) Finish(ctx context.Context) (string, error) {
	s.mu.Lock()
	if !s.finished {
		s.finished = true
		if err := s.conn.WriteMessage(websocket.BinaryMessage, saucFrame(saucTypeAudio, saucFlagLast, 0, 0, nil)); err != nil {
			s.mu.Unlock()
			return s.LastText(), fmt.Errorf("realtimevoice: byteplus asr finish: %w", err)
		}
	}
	s.mu.Unlock()

	select {
	case text := <-s.finalCh:
		return text, nil
	case <-s.readDone:
		// The server closed: its last frame, if any, already landed in finalCh.
		select {
		case text := <-s.finalCh:
			return text, nil
		default:
		}
		return s.LastText(), s.Err()
	case <-ctx.Done():
		return s.LastText(), nil
	}
}

// LastText returns the newest transcript seen so far.
func (s *ASRSession) LastText() string {
	s.lastTextMu.Lock()
	defer s.lastTextMu.Unlock()
	return s.lastText
}

// Err reports a failure that ended the stream early.
func (s *ASRSession) Err() error {
	s.errMu.Lock()
	defer s.errMu.Unlock()
	return s.err
}

// Close releases the socket. It is idempotent.
func (s *ASRSession) Close() error {
	s.mu.Lock()
	s.finished = true
	s.mu.Unlock()
	return s.conn.Close()
}

func (s *ASRSession) setErr(err error) {
	s.errMu.Lock()
	if s.err == nil {
		s.err = err
	}
	s.errMu.Unlock()
}

func (s *ASRSession) readLoop(onPartial func(text string) error) {
	defer close(s.readDone)
	for {
		msgType, data, err := s.conn.ReadMessage()
		if err != nil {
			return
		}
		if msgType != websocket.BinaryMessage {
			continue
		}
		frame, err := parseSAUC(data)
		if err != nil {
			s.setErr(err)
			return
		}
		if frame.messageType == saucTypeError {
			s.setErr(fmt.Errorf("realtimevoice: byteplus asr: %s", frame.errorText()))
			return
		}
		if frame.messageType != saucTypeServer {
			continue
		}
		text, last, err := frame.transcript()
		if err != nil {
			s.setErr(err)
			return
		}
		if text == "" {
			continue
		}
		s.lastTextMu.Lock()
		changed := text != s.lastText
		s.lastText = text
		s.lastTextMu.Unlock()
		if changed && !last && onPartial != nil {
			if err := onPartial(text); err != nil {
				s.setErr(err)
				return
			}
		}
		if last {
			select {
			case s.finalCh <- text:
			default:
			}
			return
		}
	}
}

// --- framing ---

func saucFrame(messageType, flags, serialization byte, seq uint32, payload []byte) []byte {
	header := []byte{saucVersion, messageType<<4 | flags, serialization << 4, 0x00}
	buf := make([]byte, 0, len(header)+8+len(payload))
	buf = append(buf, header...)
	if flags&saucFlagSequence != 0 {
		var seqBuf [4]byte
		binary.BigEndian.PutUint32(seqBuf[:], seq)
		buf = append(buf, seqBuf[:]...)
	}
	var sizeBuf [4]byte
	binary.BigEndian.PutUint32(sizeBuf[:], uint32(len(payload)))
	buf = append(buf, sizeBuf[:]...)
	return append(buf, payload...)
}

type saucServerFrame struct {
	messageType byte
	flags       byte
	payload     []byte
}

func parseSAUC(msg []byte) (saucServerFrame, error) {
	if len(msg) < 4 {
		return saucServerFrame{}, fmt.Errorf("realtimevoice: short SAUC frame (%d bytes)", len(msg))
	}
	flags := msg[1] & 0x0F
	off := 4
	if flags&0x04 != 0 { // event number
		off += 4
	}
	if flags&saucFlagSequence != 0 {
		off += 4
	}
	// Generation and last flags share bits; only the low two bits ever carry a
	// sequence, so the offset above is enough.
	if len(msg) < off+4 {
		return saucServerFrame{}, fmt.Errorf("realtimevoice: truncated SAUC frame")
	}
	size := int(binary.BigEndian.Uint32(msg[off : off+4]))
	off += 4
	if size < 0 || len(msg) < off+size {
		return saucServerFrame{}, fmt.Errorf("realtimevoice: SAUC payload length %d exceeds frame %d", size, len(msg))
	}
	return saucServerFrame{
		messageType: msg[1] >> 4,
		flags:       flags,
		payload:     msg[off : off+size],
	}, nil
}

func (f saucServerFrame) errorText() string {
	var e struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(f.payload, &e); err != nil {
		return string(f.payload)
	}
	return e.Error
}

// transcript decodes the running transcript and whether this frame ends the
// utterance.
func (f saucServerFrame) transcript() (string, bool, error) {
	var body struct {
		Result struct {
			Text       string `json:"text"`
			Utterances []struct {
				Text     string `json:"text"`
				Definite bool   `json:"definite"`
			} `json:"utterances"`
		} `json:"result"`
	}
	if err := json.Unmarshal(f.payload, &body); err != nil {
		return "", false, fmt.Errorf("realtimevoice: decode byteplus asr frame: %w", err)
	}
	text := body.Result.Text
	if text == "" {
		for _, u := range body.Result.Utterances {
			text += u.Text
		}
	}
	last := f.flags&saucFlagLast != 0
	for _, u := range body.Result.Utterances {
		if u.Definite {
			last = true
		}
	}
	return text, last, nil
}

// newASRHandle returns a random UUID v4 for X-Api-Connect-Id and reqid. The
// vendor only requires uniqueness per connection, but it documents the UUID
// shape, so keep it exact.
func newASRHandle() string {
	var b [16]byte
	if _, err := crand.Read(b[:]); err != nil {
		// crypto/rand only fails when the OS entropy source is broken, and a
		// colliding connect id would be rejected by the vendor with a confusing
		// error. Fall back to a nanosecond stamp, which is still unique.
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
