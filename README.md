# gfile-asr-whisperX-skill

[中文](#中文說明) | [English](#english)

---

## 中文說明

### 簡介

使用 **WhisperX** 進行語音辨識的 AI Agent Skill。支援從 Google Drive 下載音訊/影片檔案，並使用 WhisperX（faster-whisper + wav2vec2 對齊 + 說話者辨識）在本地 GPU 上進行轉逐字稿。

### WhisperX vs 基本 Whisper/speaches

| 功能 | 基本 Whisper/speaches | WhisperX |
|------|---------------------|----------|
| VAD 預處理 | 內建但不可調 | Silero VAD，可調閾值 |
| 時間戳精度 | segment 級別 | **word 級別**（wav2vec2 對齊）|
| 說話者辨識 | 無 | **有**（pyannote） |
| 批次推理 | 無 | **有**（~70x 即時速度）|
| 幻覺問題 | 常見 | 極少 |

### 前置需求

| 元件 | 版本 | 用途 |
|------|------|------|
| Python | 3.12（**不支援 3.14**） | WhisperX + PyTorch |
| NVIDIA GPU | 任何支援 CUDA 的顯卡 | GPU 加速 |
| NVIDIA Driver | 550+ | RTX 50 系列需要最新驅動 |
| ffmpeg | v5+ | 音訊轉換 |
| gdown | 最新版 | Google Drive 下載 |

> **RTX 50 系列（Blackwell/sm_120）注意事項：** pip install whisperx 會自動安裝相容的 PyTorch + CUDA 12.8。

### 安裝

#### 1. 建立 Python 虛擬環境（使用 Python 3.12）

```bash
/usr/bin/python3.12 -m venv /path/to/asr-venv
```

#### 2. 安裝 WhisperX（一鍵安裝所有依賴）

```bash
/path/to/asr-venv/bin/pip install "whisperx>=3.3.4" gdown silero-vad
```

這會自動安裝 PyTorch、faster-whisper、pyannote-audio、wav2vec2 等所有依賴。

#### 3. 安裝系統工具

```bash
sudo apt install -y ffmpeg
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
/path/to/whisperx-venv/bin/python3 scripts/transcribe_whisperx.py /path/to/audio.mp3 --lang zh

# 含說話者辨識
HF_TOKEN=hf_xxx /path/to/whisperx-venv/bin/python3 scripts/transcribe_whisperx.py /path/to/audio.mp3 --lang zh --diarize

# 指定輸出格式
/path/to/whisperx-venv/bin/python3 scripts/transcribe_whisperx.py /path/to/audio.mp3 --lang zh --format json
```

#### AI Agent

在聊天中傳送 Google Drive 連結並說「轉逐字稿」、「transcribe」等關鍵字。

### AI Agent 平台安裝

**OpenClaw：**
```bash
ln -sf /path/to/gfile-asr-whisperX-skill ~/.openclaw/skills/gfile-asr-whisperx
```

**Cursor / Claude Code / Codex / Gemini CLI：**
將此 repo clone 到專案目錄，agent 會自動讀取 `SKILL.md` / `CLAUDE.md` / `AGENTS.md` / `GEMINI.md`。

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

### Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.12 (**not 3.14**) | WhisperX + PyTorch |
| NVIDIA GPU | Any with CUDA | GPU acceleration |
| NVIDIA Driver | 550+ | RTX 50 series needs latest |
| ffmpeg | v5+ | Audio conversion |
| gdown | Latest | Google Drive downloads |

> **RTX 50 series (Blackwell/sm_120) note:** `pip install whisperx` automatically installs compatible PyTorch + CUDA 12.8.

### Installation

#### 1. Create venv with Python 3.12

```bash
/usr/bin/python3.12 -m venv /path/to/asr-venv
```

#### 2. Install WhisperX (one command installs all deps)

```bash
/path/to/asr-venv/bin/pip install "whisperx>=3.3.4" gdown silero-vad
```

This automatically installs PyTorch, faster-whisper, pyannote-audio, wav2vec2, and all other dependencies.

#### 3. (Optional) HuggingFace token for speaker diarization

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

**OpenClaw:**
```bash
ln -sf /path/to/gfile-asr-whisperX-skill ~/.openclaw/skills/gfile-asr-whisperx
```

**Cursor / Claude Code / Codex / Gemini CLI:**
Clone this repo into your project directory.

## License

MIT
