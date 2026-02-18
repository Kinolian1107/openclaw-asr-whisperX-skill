---
name: gfile-asr-whisperx
description: >
  Download audio/video files from Google Drive and transcribe using WhisperX
  (faster-whisper + wav2vec2 alignment + speaker diarization). GPU-accelerated.
  Triggers on keywords: 轉逐字稿, 轉文字, transcribe, transcript, 語音轉文字, ASR, 字幕, subtitle.
metadata:
  openclaw:
    emoji: "🎙️"
    requires:
      bins: ["ffmpeg", "gdown", "python3"]
    os: ["linux"]
---

# Google Drive ASR — WhisperX Mode

Transcribe audio/video from Google Drive using **WhisperX** (faster-whisper + wav2vec2 alignment + speaker diarization).

## How It Works

1. **WhisperX** loads faster-whisper with built-in VAD (Silero) — no hallucinations
2. **Batch inference** for maximum GPU throughput
3. **wav2vec2 forced alignment** for word-level accurate timestamps
4. **Optional speaker diarization** (identifies who said what)

## Trigger Conditions

Same as gfile-asr-speaches. Activate when user sends Google Drive link + mentions transcription keywords.

## Prerequisites

Python venv at `/home/kino/asr/.venv-whisperx/` with whisperx + PyTorch nightly (CUDA 12.8 for RTX 50 series):

```bash
/usr/bin/python3.12 -m venv /home/kino/asr/.venv-whisperx
/home/kino/asr/.venv-whisperx/bin/pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
/home/kino/asr/.venv-whisperx/bin/pip install whisperx gdown
/home/kino/asr/.venv-whisperx/bin/pip install --pre nvidia-cudnn-cu12 --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --no-deps
```

## Workflow

**CRITICAL: Run ALL steps without stopping. Deliver results via Telegram when done.**

### Step 1: Download from Google Drive

```bash
gdown "https://drive.google.com/uc?id={FILE_ID}" -O /home/kino/asr/{filename}
```

### Step 2: Run WhisperX Transcription

```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/transcribe_whisperx.py" \
    /home/kino/asr/{filename} --lang zh --format srt
```

For speaker diarization (requires HF_TOKEN env var):

```bash
HF_TOKEN=hf_xxx /home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/transcribe_whisperx.py" \
    /home/kino/asr/{filename} --lang zh --format srt --diarize
```

### Step 3: Report Results & Deliver via Telegram

1. Copy SRT to workspace:
   ```bash
   cp /home/kino/asr/{basename}.srt /home/kino/.openclaw/workspace/{basename}.srt
   ```

2. Send via Telegram `message` tool:
   ```
   action: send
   message: "轉寫完成！{basename}.srt（WhisperX，{duration}s）"
   filePath: /home/kino/.openclaw/workspace/{basename}.srt
   ```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WHISPERX_MODEL` | `large-v3-turbo` | Model size |
| `ASR_DEVICE` | `cuda` | Device (cuda/cpu) |
| `ASR_COMPUTE_TYPE` | `int8` | Compute type |
| `WHISPERX_BATCH_SIZE` | `16` | Batch size for inference |
| `HF_TOKEN` | (none) | HuggingFace token for diarization |
| `HF_HOME` | `/home/kino/ollama-models/huggingface-hub` | Model cache |

## Supported Input

- **Audio**: MP3, WAV, M4A, FLAC, OGG, AAC, WMA
- **Video**: MP4, MKV, AVI, MOV, WebM, FLV
- **Sources**: Google Drive links, local file paths

## References

- [WhisperX](https://github.com/m-bain/whisperX) — INTERSPEECH 2023
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
