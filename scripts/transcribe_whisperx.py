#!/usr/bin/env python3
"""
WhisperX ASR Pipeline: Google Drive → ffmpeg → WhisperX → SRT

Uses WhisperX for:
- Built-in VAD preprocessing (eliminates hallucinations)
- wav2vec2 forced alignment (word-level timestamp accuracy)
- Optional speaker diarization (identifies who is speaking)
- Batched inference (~70x realtime on GPU)

Usage:
    python3 transcribe_whisperx.py <input_file> [--lang zh] [--format srt] [--diarize]
    python3 transcribe_whisperx.py <gdrive_url> [--lang zh] [--format srt]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ASR_DIR = os.environ.get("ASR_DIR", "/home/kino/asr")
HF_CACHE = os.environ.get("HF_HOME", "/home/kino/ollama-models/huggingface-hub")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
COMPUTE_TYPE = os.environ.get("WHISPERX_COMPUTE_TYPE", "int8")
MODEL_SIZE = os.environ.get("WHISPERX_MODEL", "large-v3-turbo")
BATCH_SIZE = int(os.environ.get("WHISPERX_BATCH_SIZE", "16"))
GDOWN_PATH = os.environ.get("GDOWN_PATH", "gdown")


def run_cmd(cmd, **kwargs):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def detect_best_device():
    """Auto-detect usable device: try CUDA, fall back to CPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        try:
            t = torch.zeros(1, device="cuda")
            del t
            torch.cuda.empty_cache()
            return "cuda"
        except Exception:
            print("WARNING: CUDA available but kernel execution failed (GPU too new for this PyTorch build)")
            print("         Falling back to CPU. This will be slower but still works.")
            return "cpu"
    except ImportError:
        return "cpu"


def extract_gdrive_id(url: str) -> str:
    patterns = [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return ""


def download_gdrive(url: str, output_dir: str) -> str:
    file_id = extract_gdrive_id(url)
    if not file_id:
        raise ValueError(f"Cannot extract Google Drive file ID from: {url}")
    output = os.path.join(output_dir, f"gdrive_{file_id}")
    print(f"Downloading from Google Drive (file_id: {file_id})...")
    stdout, stderr, rc = run_cmd(
        f'{GDOWN_PATH} "https://drive.google.com/uc?id={file_id}" -O "{output}"'
    )
    if rc != 0:
        raise RuntimeError(f"gdown failed: {stderr}")
    ext = detect_extension(output)
    if ext:
        new_path = f"{output}.{ext}"
        os.rename(output, new_path)
        return new_path
    return output


def detect_extension(path: str) -> str:
    stdout, _, _ = run_cmd(f'file --brief --mime-type "{path}"')
    mime_map = {
        "audio/mpeg": "mp3", "audio/mp4": "m4a", "audio/x-wav": "wav",
        "audio/wav": "wav", "audio/flac": "flac", "audio/ogg": "ogg",
        "video/mp4": "mp4", "video/x-matroska": "mkv",
        "video/quicktime": "mov", "video/webm": "webm",
    }
    return mime_map.get(stdout.strip(), "")


def detect_mime(filepath: str) -> str:
    stdout, _, _ = run_cmd(f'file --brief --mime-type "{filepath}"')
    return stdout.strip()


def get_duration(filepath: str) -> float:
    stdout, _, _ = run_cmd(
        f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{filepath}"'
    )
    return float(stdout.strip())


def extract_audio(input_path: str, output_path: str):
    print(f"Extracting audio → {output_path}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", output_path],
        capture_output=True, check=True
    )


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, filepath, include_speaker=False):
    with open(filepath, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            text = seg["text"].strip()
            if include_speaker and "speaker" in seg:
                text = f"[{seg['speaker']}] {text}"
            f.write(f"{text}\n\n")


def transcribe(audio_path: str, language: str, output_format: str,
               diarize: bool = False, output_dir: str = ASR_DIR,
               device: str = None):
    import whisperx
    import torch

    os.environ["HF_HOME"] = HF_CACHE
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

    if device is None:
        device = detect_best_device()

    compute = COMPUTE_TYPE if device == "cuda" else "int8"
    print(f"Loading WhisperX model: {MODEL_SIZE} (device={device}, compute={compute})")

    model = whisperx.load_model(
        MODEL_SIZE,
        device=device,
        compute_type=compute,
        download_root=os.path.join(HF_CACHE, "whisperx"),
    )

    print(f"Loading audio: {audio_path}")
    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / 16000
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")

    print(f"Transcribing with batch_size={BATCH_SIZE}...")
    result = model.transcribe(
        audio,
        batch_size=BATCH_SIZE,
        language=language if language != "auto" else None,
    )

    detected_lang = result.get("language", language)
    print(f"Detected language: {detected_lang}")
    print(f"Initial segments: {len(result['segments'])}")

    print("Aligning with wav2vec2 for precise timestamps...")
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        print(f"Aligned segments: {len(result['segments'])}")
    except Exception as e:
        print(f"WARNING: Alignment failed ({e}), using original timestamps")

    if diarize and HF_TOKEN:
        print("Running speaker diarization...")
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=HF_TOKEN,
                device=device,
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            speakers = set(s.get("speaker", "") for s in result["segments"] if s.get("speaker"))
            print(f"Identified {len(speakers)} speakers: {', '.join(sorted(speakers))}")
        except Exception as e:
            print(f"WARNING: Diarization failed ({e}), continuing without speaker labels")
    elif diarize and not HF_TOKEN:
        print("WARNING: HF_TOKEN not set, skipping diarization.")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    segments = result["segments"]
    basename = Path(audio_path).stem

    srt_path = os.path.join(output_dir, f"{basename}.srt")
    write_srt(segments, srt_path, include_speaker=diarize)
    print(f"SRT: {srt_path}")

    txt_path = os.path.join(output_dir, f"{basename}.txt")
    full_text_parts = []
    for seg in segments:
        text = seg["text"].strip()
        if diarize and "speaker" in seg:
            text = f"[{seg['speaker']}] {text}"
        full_text_parts.append(text)
    full_text = "\n".join(full_text_parts)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"TXT: {txt_path}")

    if output_format in ("json", "all"):
        json_path = os.path.join(output_dir, f"{basename}.json")
        json_segments = []
        for seg in segments:
            entry = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            if "speaker" in seg:
                entry["speaker"] = seg["speaker"]
            if "words" in seg:
                entry["words"] = [
                    {"word": w.get("word", ""), "start": w.get("start", 0), "end": w.get("end", 0)}
                    for w in seg["words"] if "word" in w
                ]
            json_segments.append(entry)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "text": full_text,
                "segments": json_segments,
                "duration": duration,
                "language": detected_lang,
                "model": MODEL_SIZE,
                "engine": "whisperx",
                "device": device,
                "diarization": diarize,
            }, f, ensure_ascii=False, indent=2)
        print(f"JSON: {json_path}")

    print(f"\n{'='*50}")
    print(f"Transcription complete!")
    print(f"  Duration: {duration:.0f}s ({duration/60:.1f} min)")
    print(f"  Segments: {len(segments)}")
    print(f"  Language: {detected_lang}")
    print(f"  Engine: WhisperX ({MODEL_SIZE}) on {device}")
    if diarize:
        speakers = set(s.get("speaker", "") for s in segments if s.get("speaker"))
        print(f"  Speakers: {len(speakers)}")
    print(f"  SRT: {srt_path}")

    return segments


def main():
    parser = argparse.ArgumentParser(
        description="WhisperX ASR Pipeline with alignment and optional diarization")
    parser.add_argument("input", help="Input audio/video file path or Google Drive URL")
    parser.add_argument("--lang", default="zh", help="Language code (zh, en, ja, auto)")
    parser.add_argument("--format", default="srt", choices=["srt", "text", "json", "all"],
                        help="Output format (default: srt)")
    parser.add_argument("--diarize", action="store_true",
                        help="Enable speaker diarization (requires HF_TOKEN)")
    parser.add_argument("--output-dir", default=ASR_DIR, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for inference")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="Force device (default: auto-detect)")
    args = parser.parse_args()

    input_path = args.input

    if "drive.google.com" in input_path:
        input_path = download_gdrive(input_path, args.output_dir)

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    mime = detect_mime(input_path)
    print(f"Input: {input_path} ({mime})")

    basename = Path(input_path).stem
    wav_path = os.path.join(args.output_dir, f"{basename}.wav")

    if mime.startswith("video/") or (mime.startswith("audio/") and not input_path.endswith(".wav")):
        extract_audio(input_path, wav_path)
    elif input_path.endswith(".wav"):
        wav_path = input_path

    segments = transcribe(
        wav_path,
        language=args.lang,
        output_format=args.format,
        diarize=args.diarize,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(f"\nDone! {len(segments)} segments transcribed.")


if __name__ == "__main__":
    main()
