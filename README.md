# gfile-asr-whisperX-skill

[中文](#中文說明) | [English](#english)

---

## 中文說明

### 簡介

使用 **WhisperX** 進行語音辨識的 AI Agent Skill。支援從 Google Drive 下載音訊/影片檔案，並使用 WhisperX（faster-whisper + wav2vec2 對齊 + 說話者辨識）在本地 GPU 上進行轉逐字稿。

### WhisperX vs 基本 Whisper

| 功能 | 基本 Whisper/speaches | WhisperX |
|------|---------------------|----------|
| VAD 預處理 | 內建但不可調 | Silero VAD，可調閾值 |
| 時間戳精度 | segment 級別 | **word 級別**（wav2vec2 對齊）|
| 說話者辨識 | 無 | **有**（pyannote） |
| 批次推理 | 無 | **有**（~70x 即時速度）|
| 幻覺問題 | 常見 | 極少 |

### 安裝

#### 1. 建立 Python 虛擬環境

```bash
python3 -m venv /home/kino/asr/.venv
source /home/kino/asr/.venv/bin/activate
```

#### 2. 安裝依賴

```bash
pip install whisperx torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install gdown
```

#### 3. 安裝系統工具

```bash
# ffmpeg
brew install ffmpeg  # 或 apt install ffmpeg

# gdown (Google Drive 下載)
pip install gdown
```

#### 4.（選用）設定 HuggingFace Token（說話者辨識需要）

```bash
export HF_TOKEN=hf_your_token_here
```

需要到 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 同意使用條款。

### 使用方式

#### 命令列

```bash
# 基本轉逐字稿
/home/kino/asr/.venv/bin/python3 scripts/transcribe_whisperx.py /path/to/audio.mp3 --lang zh

# 含說話者辨識
HF_TOKEN=hf_xxx /home/kino/asr/.venv/bin/python3 scripts/transcribe_whisperx.py /path/to/audio.mp3 --lang zh --diarize

# 指定輸出格式
/home/kino/asr/.venv/bin/python3 scripts/transcribe_whisperx.py /path/to/audio.mp3 --lang zh --format json
```

#### AI Agent

在聊天中傳送 Google Drive 連結並說「轉逐字稿」、「transcribe」等關鍵字。

### AI Agent 平台安裝

#### OpenClaw

```bash
ln -sf /path/to/gfile-asr-whisperX-skill ~/.openclaw/skills/gfile-asr-whisperx
```

#### Cursor / Claude Code / Codex

將 `SKILL.md`（或 `CLAUDE.md` / `AGENTS.md`）加入專案根目錄。

### 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `WHISPERX_MODEL` | `large-v3-turbo` | Whisper 模型大小 |
| `ASR_DEVICE` | `cuda` | 裝置（cuda/cpu）|
| `ASR_COMPUTE_TYPE` | `int8` | 計算類型 |
| `WHISPERX_BATCH_SIZE` | `16` | 批次推理大小 |
| `HF_TOKEN` | (無) | HuggingFace token（說話者辨識用）|
| `HF_HOME` | `/home/kino/ollama-models/huggingface-hub` | 模型快取目錄 |
| `ASR_DIR` | `/home/kino/asr` | 工作目錄 |

---

## English

### Overview

AI Agent Skill for speech recognition using **WhisperX**. Downloads audio/video from Google Drive and transcribes locally using WhisperX (faster-whisper + wav2vec2 alignment + speaker diarization) with GPU acceleration.

### Installation

#### 1. Create Python virtual environment

```bash
python3 -m venv /home/kino/asr/.venv
source /home/kino/asr/.venv/bin/activate
```

#### 2. Install dependencies

```bash
pip install whisperx torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install gdown
```

#### 3. Install system tools

```bash
brew install ffmpeg  # or apt install ffmpeg
```

#### 4. (Optional) Set HuggingFace Token for speaker diarization

```bash
export HF_TOKEN=hf_your_token_here
```

### Usage

```bash
# Basic transcription
python3 scripts/transcribe_whisperx.py /path/to/audio.mp3 --lang zh

# With speaker diarization
HF_TOKEN=hf_xxx python3 scripts/transcribe_whisperx.py /path/to/audio.mp3 --lang zh --diarize
```

### AI Agent Installation

#### OpenClaw
```bash
ln -sf /path/to/gfile-asr-whisperX-skill ~/.openclaw/skills/gfile-asr-whisperx
```

### License

MIT
