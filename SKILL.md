---
name: gfile-asr-whisperx
description: >
  Download audio/video files from Google Drive and transcribe using WhisperX
  (faster-whisper + wav2vec2 alignment + speaker diarization). GPU-accelerated.
  Features: OpenCC 繁體輸出, hotwords, corrections, speaker embedding matching.
  Triggers on keywords: 轉逐字稿, 轉文字, transcribe, transcript, 語音轉文字, ASR, 字幕, subtitle,
  辨識成文字, 語音辨識.
metadata:
  openclaw:
    emoji: "🎙️"
    requires:
      bins: ["ffmpeg", "gdown", "python3"]
    os: ["linux"]
---

# Google Drive ASR — WhisperX Mode (v2)

Transcribe audio/video from Google Drive using **WhisperX** (faster-whisper + wav2vec2 alignment + speaker diarization).

## v2 Features

- **Topic-guided initial_prompt** — improves accuracy for domain-specific content
- **Audio denoising** — optional ffmpeg-based noise reduction
- **OpenCC s2twp** — auto-converts simplified → traditional Chinese (Taiwan usage)
- **Hotwords** — faster-whisper native hotword boosting from `whisperx_hotwords.txt`
- **Corrections dictionary** — post-processing replacements from `asr_corrections.json`
- **Speaker embedding** — auto-extract speaker samples, match against registered DB
- **Speaker diarization** — pyannote speaker-diarization-3.1

## Mode Check

**Before executing, read `/home/kino/.openclaw/workspace/asr_config.json`.**
- If `"mode": "whisperx"` → proceed with this skill
- If `"mode": "speaches"` → use the `gfile-asr-speaches` skill instead
- User can switch modes with `/asrmode`

## Trigger Conditions

Activate when ANY of the following are true:

1. User sends a **Google Drive link** + mentions: 轉逐字稿, 轉文字, transcribe, transcript, 語音轉文字, ASR, 字幕, subtitle, 摘要, summary, 分析, 辨識成文字, 語音辨識
2. User provides a **local file path** to audio/video and asks for transcription
3. User says "transcribe" or "轉逐字稿" referencing a previously downloaded file

## Pre-Transcription Interaction (IMPORTANT)

**Before starting transcription, check if the user has provided enough context:**

### 1. Topic / 主題
If the user did NOT mention the audio topic/subject, ask:
```
這個音檔的主題是什麼？（例如：財經討論、會議記錄、課堂講座、日常對話等）
提供主題可以提升辨識準確度 📈
如果不確定，直接說「不用」我就開始轉了。
```
Use the user's answer as `--topic` parameter.

### 2. Denoising / 降噪
If the user explicitly mentions 降噪、雜音多、音質不好、背景噪音, add `--denoise` flag.
**Do NOT proactively ask about denoising** — only apply when user mentions it.

### 3. Hotwords / 熱詞
When the user says 「增加熱詞」、「加入熱詞」、「新增 hotword」 or similar:
- Append the new word(s) to `/home/kino/.openclaw/workspace/whisperx_hotwords.txt` (one per line)
- Confirm: "已新增熱詞：XXX ✅ 下次轉逐字稿時會自動使用。"

The script automatically loads hotwords from `whisperx_hotwords.txt` every run.
You can tell the user: "如果有專有名詞想加強辨識，可以跟我說「增加熱詞 XXX」"

### 4. Speaker Diarization
If user asks to identify speakers (辨識說話者, 分辨講者, diarize, 誰在說話), add `--diarize`.

## Prerequisites

Python venv at `/home/kino/asr/.venv-whisperx/` with whisperx + PyTorch nightly (CUDA 12.8 for RTX 50 series).

Required packages: `whisperx`, `gdown`, `opencc-python-reimplemented`, `soundfile`, `numpy`, `pyannote.audio`

## Workflow

**CRITICAL: Run ALL steps without stopping. Deliver results via Telegram when done.**

### Step 1: Download from Google Drive

```bash
gdown "https://drive.google.com/uc?id={FILE_ID}" -O /home/kino/asr/{filename}
```

### Step 2: Run WhisperX Transcription

Basic (with topic):
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/transcribe_whisperx.py" \
    /home/kino/asr/{filename} --lang zh --format srt --topic "主題描述"
```

With denoising:
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/transcribe_whisperx.py" \
    /home/kino/asr/{filename} --lang zh --format srt --topic "主題" --denoise
```

With speaker diarization:
```bash
HF_TOKEN=hf_xxx /home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/transcribe_whisperx.py" \
    /home/kino/asr/{filename} --lang zh --format srt --topic "主題" --diarize
```

With subtitle splitting (limit characters per line):
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/transcribe_whisperx.py" \
    /home/kino/asr/{filename} --lang zh --format srt --topic "主題" --max-chars 20
```

Full options:
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/transcribe_whisperx.py" \
    /home/kino/asr/{filename} \
    --lang zh \
    --format srt \
    --topic "主題描述" \
    --denoise \
    --diarize \
    --max-chars 20 \
    --hotwords-file /home/kino/.openclaw/workspace/whisperx_hotwords.txt \
    --corrections-file /home/kino/.openclaw/workspace/asr_corrections.json
```

The script automatically:
- Loads hotwords from `whisperx_hotwords.txt` (boosts accuracy for domain terms)
- Loads corrections from `asr_corrections.json` (fixes known ASR errors)
- Converts output to traditional Chinese via OpenCC (s2twp mode)
- When `--diarize`: extracts speaker audio samples → matches against speaker DB → saves unknown speakers for future matching

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

3. If `--diarize` was used and there are unmatched speakers, inform the user:
   ```
   辨識出 {n} 位說話者。
   未匹配的說話者音檔已保存在：{speaker_samples_dir}
   你可以之後告訴我「把 SPEAKER_00 命名為 XXX」來註冊聲紋。
   ```

## Speaker Embedding Management

### Registering a speaker (user uploads audio + provides name)

When user says "註冊說話者"、"上傳聲紋"、"register speaker" etc:
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/speaker_embed.py" \
    register --name "名字" --audio /path/to/audio.wav
```

### Renaming a SPEAKER_XX from a previous session

When user says "把 SPEAKER_00 命名為 XXX" or "rename speaker":
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/speaker_embed.py" \
    rename --sample-dir /home/kino/asr/speaker_samples/{session_dir} \
    --speaker SPEAKER_00 --name "名字"
```

### Listing registered speakers
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/speaker_embed.py" list
```

### Deleting a registered speaker
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/speaker_embed.py" \
    delete --name "名字"
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WHISPERX_MODEL` | `large-v3-turbo` | Model size |
| `WHISPERX_DEVICE` | `auto` | Device (cuda/cpu/auto) |
| `ASR_COMPUTE_TYPE` | `int8` | Compute type |
| `WHISPERX_BATCH_SIZE` | `16` | Batch size for inference |
| `HF_TOKEN` | (none) | HuggingFace token for diarization |
| `HF_HOME` | `/home/kino/ollama-models/huggingface-hub` | Model cache |
| `--topic` | (none) | Topic description for initial_prompt |
| `--denoise` | false | Apply audio denoising |
| `--no-opencc` | false | Disable OpenCC traditional Chinese conversion |
| `--max-chars` | 0 (disabled) | Max characters per subtitle segment (recommended: 20 for Chinese) |

## Config Files

| File | Location | Purpose |
|------|----------|---------|
| `asr_config.json` | `/home/kino/.openclaw/workspace/` | ASR mode & settings |
| `whisperx_hotwords.txt` | `/home/kino/.openclaw/workspace/` | Hotword list (one per line) |
| `asr_corrections.json` | `/home/kino/.openclaw/workspace/` | Error→correct word mappings |
| `speakers.json` | `/home/kino/asr/speaker_embeddings/` | Registered speaker metadata |

## Supported Input

- **Audio**: MP3, WAV, M4A, FLAC, OGG, AAC, WMA
- **Video**: MP4, MKV, AVI, MOV, WebM, FLV
- **Sources**: Google Drive links, local file paths

## /asrmode Command

When user types `/asrmode`:

1. Read `/home/kino/.openclaw/workspace/asr_config.json`
2. Show current mode and options with inline buttons
3. After user selects, update `asr_config.json` `"mode"` field and confirm

## References

- [WhisperX](https://github.com/m-bain/whisperX) — INTERSPEECH 2023
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — hotwords support in v1.2+
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [OpenCC](https://github.com/BYVoid/OpenCC) — Chinese conversion
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) — speaker diarization & embedding
