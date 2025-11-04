# config.py
# All configuration constants for the application

import torch

# --- Audio & VAD ---
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHUNK_DURATION = 0.03  # 30ms for VAD frames
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
VAD_MODE = 2  # 0-3, aggressiveness of VAD
FRAME_DURATION_MS = 30  # WebRTC VAD frame size in ms (10, 20, or 30)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Speech Detection Parameters ---
MIN_SPEECH_DURATION = 0.5  # Minimum speech duration (seconds)
MIN_SPEECH_FRAMES = int(MIN_SPEECH_DURATION * SAMPLE_RATE)
MAX_SILENCE_DURATION = 0.6  # Max silence before ending speech (seconds)
MAX_SILENCE_FRAMES = int(MAX_SILENCE_DURATION * SAMPLE_RATE)

# --- Concurrency Parameters ---
MAX_SPEECH_DURATION = 5.0 # Max speech before forcing transcript (seconds)
MAX_SPEECH_FRAMES = int(MAX_SPEECH_DURATION * SAMPLE_RATE)
OVERLAP_DURATION = 1.0 # Overlap for concurrent chunks (seconds)
OVERLAP_FRAMES = int(OVERLAP_DURATION * SAMPLE_RATE)