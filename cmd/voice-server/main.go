// Command voice-server serves the Gemini Live (BidiGenerateContent)
// WebSocket protocol backed by a scripted pipeline.
//
// It is the protocol conformance harness for the STT + text-LLM + TTS
// pipeline: a client that works against Gemini should connect unchanged and
// complete a turn, receive audio and transcripts, barge in, and round-trip a
// tool call. Swap ScriptedEngine for a real pipeline later; the wire behavior
// is already pinned by tests.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/dinghaoz/realtimify"
)

func main() {
	var (
		listen     = flag.String("listen", ":8090", "listen address")
		apiKey     = flag.String("api-key", "", "require this ?key= / Bearer token (empty = accept any)")
		reply      = flag.String("reply", "This is the scripted mock replying.", "assistant reply text")
		userText   = flag.String("user-transcript", "Hello, this is the user.", "transcript reported for the user's turn")
		audioMS    = flag.Int("audio-ms", 1200, "reply audio duration in milliseconds")
		chunkMS    = flag.Int("chunk-ms", 20, "audio frame size in milliseconds")
		ttfbMS     = flag.Int("ttfb-ms", 250, "delay before the first assistant output")
		scriptPath = flag.String("script", "", "JSON script file (overrides -reply/-user-transcript)")
		loop       = flag.Bool("loop", false, "replay the script instead of repeating its last turn")
		verbose    = flag.Bool("v", false, "debug logging")

		llmKey      = flag.String("llm-key", os.Getenv("VERTEX_API_KEY"), "Gemini API key; with -asr-key enables the live pipeline")
		llmProject  = flag.String("llm-project", os.Getenv("VERTEX_PROJECT_ID"), "Vertex project id")
		llmModel    = flag.String("llm-model", "gemini-2.5-flash", "text LLM model")
		asrKey      = flag.String("asr-key", os.Getenv("BYTEPLUS_API_KEY"), "BytePlus Seed Speech API key (ASR + TTS)")
		ttsKey      = flag.String("tts-key", "", "BytePlus TTS key; defaults to -asr-key")
		voice       = flag.String("voice", "zh_female_vv_uranus_bigtts", "BytePlus voice type (音色)")
		asrLanguage = flag.String("asr-language", "", "ASR language hint (empty = auto)")
		ttsLanguage = flag.String("tts-language", "", "TTS language hint (empty = auto)")
		ttsModel    = flag.String("tts-model", "", "TTS model (empty = account default)")
		temperature = flag.Float64("temperature", 0.6, "LLM sampling temperature")
		maxTokens   = flag.Int("max-tokens", 0, "LLM max output tokens (0 = provider default)")
		clauseRunes = flag.Int("clause-runes", 40, "max runes per TTS clause before an early cut")
		system      = flag.String("system", "你是一个实时语音助手。回答口语化、简短，最多两句话。", "fallback system instruction")
	)
	flag.Parse()

	level := slog.LevelInfo
	if *verbose {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	script, err := loadScript(*scriptPath, *userText, *reply, *audioMS, *chunkMS, *ttfbMS, *loop)
	if err != nil {
		logger.Error("load script", "err", err)
		os.Exit(2)
	}

	if *ttsKey == "" {
		*ttsKey = *asrKey
	}
	live := *llmKey != "" && *asrKey != ""
	if live && *llmProject == "" {
		logger.Error("live pipeline needs -llm-project (or VERTEX_PROJECT_ID)")
		os.Exit(2)
	}

	srv := &realtimify.Server{
		APIKey: *apiKey,
		Logger: logger,
		Engine: func(_ context.Context, setup *realtimify.Setup) (realtimify.Engine, error) {
			voiceName := ""
			if setup.GenerationConfig != nil && setup.GenerationConfig.SpeechConfig != nil &&
				setup.GenerationConfig.SpeechConfig.VoiceConfig != nil &&
				setup.GenerationConfig.SpeechConfig.VoiceConfig.PrebuiltVoiceConfig != nil {
				voiceName = setup.GenerationConfig.SpeechConfig.VoiceConfig.PrebuiltVoiceConfig.VoiceName
			}
			logger.Info("session setup", "model", setup.Model, "gemini_voice", voiceName,
				"tools", len(setup.Tools), "live", live)
			if !live {
				return realtimify.NewScriptedEngine(script), nil
			}

			instruction := *system
			if setup.SystemInstruction != nil {
				var b strings.Builder
				for _, part := range setup.SystemInstruction.Parts {
					b.WriteString(part.Text)
				}
				if text := strings.TrimSpace(b.String()); text != "" {
					instruction = text
				}
			}
			asr := &realtimify.BytePlusASR{APIKey: *asrKey, Language: *asrLanguage}
			// The model travels per request (LLMOptions.Model), so the provider
			// keeps no model of its own here.
			llm := &realtimify.GeminiLLM{
				Project: *llmProject, Location: "global", APIKey: *llmKey,
				// Realtime voice cannot absorb default thinking latency or cost.
				ThinkingBudget: new(int),
			}
			tts := &realtimify.BytePlusTTS{APIKey: *ttsKey}
			return realtimify.NewPipeline(realtimify.Config{
				ASR:        asr,
				LLM:        llm,
				TTS:        tts,
				ASROptions: realtimify.ASROptions{Language: *asrLanguage},
				TTSOptions: realtimify.TTSOptions{Voice: *voice, Language: *ttsLanguage, Model: *ttsModel},
				LLMOptions: realtimify.LLMOptions{
					Model:             *llmModel,
					SystemInstruction: instruction,
					Temperature:       *temperature,
					MaxOutputTokens:   *maxTokens,
				},
				Tools:          setup.Tools,
				MaxClauseRunes: *clauseRunes,
			}), nil
		},
	}

	mux := http.NewServeMux()
	// Vertex AI path (production uses this one).
	mux.Handle("/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent", srv)
	// AI Studio (generativelanguage) path.
	mux.Handle("/ws/google.ai.generativelanguage.v1beta1.GenerativeService/BidiGenerateContent", srv)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
	})

	httpSrv := &http.Server{
		Addr:              *listen,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		logger.Info("voice-server listening", "addr", *listen,
			"turns", len(script.Turns), "api_key_required", *apiKey != "")
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("listen", "err", err)
			os.Exit(1)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpSrv.Shutdown(ctx)
}

func loadScript(path, userText, reply string, audioMS, chunkMS, ttfbMS int, loop bool) (realtimify.Script, error) {
	if path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			return realtimify.Script{}, err
		}
		var script realtimify.Script
		if err := json.Unmarshal(data, &script); err != nil {
			return realtimify.Script{}, err
		}
		if script.ChunkMS == 0 {
			script.ChunkMS = chunkMS
		}
		if script.TTFBMS == 0 {
			script.TTFBMS = ttfbMS
		}
		return script, nil
	}
	return realtimify.Script{
		ChunkMS: chunkMS,
		TTFBMS:  ttfbMS,
		Loop:    loop,
		Turns: []realtimify.ScriptTurn{{
			UserTranscript: userText,
			AssistantText:  reply,
			AudioMS:        audioMS,
		}},
	}, nil
}
