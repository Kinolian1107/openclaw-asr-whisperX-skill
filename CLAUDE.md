# WhisperX ASR Skill

When asked to transcribe audio/video (keywords: 轉逐字稿, 轉文字, transcribe, ASR, 字幕):

**CRITICAL: Run the entire pipeline without stopping. Do not wait for user input between steps.**

1. Download from Google Drive: `gdown "https://drive.google.com/uc?id={FILE_ID}" -O /home/kino/asr/{filename}`
2. Run WhisperX: `/home/kino/asr/.venv-whisperx/bin/python3 scripts/transcribe_whisperx.py /home/kino/asr/{filename} --lang zh --format srt`
3. Copy SRT to workspace and deliver via messaging tool

Venv: `/home/kino/asr/.venv-whisperx/` (Python 3.12, PyTorch nightly cu128, WhisperX 3.8.1)
Model cache: `/home/kino/ollama-models/huggingface-hub`
Working dir: `/home/kino/asr`
