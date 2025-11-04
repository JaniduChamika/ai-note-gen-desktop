# asr_engine.py
# Contains all logic for audio processing and transcription

import numpy as np
import torch
from transformers import pipeline
import webrtcvad
import noisereduce as nr
import queue
import threading
import time

# Import all constants from our config file
from config import *


class AsrEngine:
    def __init__(self, gui_queue):
        # Queues & State
        self.gui_queue = gui_queue
        self.transcription_queue = queue.Queue()
        self.stop_event = threading.Event()

        # ASR Model & VAD
        self.pipe = None
        self.vad = webrtcvad.Vad(VAD_MODE)

        # Audio Buffers
        self.audio_buffer = []
        self.silence_counter = 0
        self.is_speaking = False

    def reset(self):
        """Resets the audio buffers and state."""
        self.audio_buffer = []
        self.silence_counter = 0
        self.is_speaking = False
        self.stop_event.clear()

    def stop(self):
        """Signals all threads to stop."""
        self.stop_event.set()

    def is_speech_detected(self, chunk):
        """Detect speech in a chunk using WebRTC VAD."""
        if len(chunk) != FRAME_SIZE:
            return False
        try:
            audio_int16 = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
            frame_bytes = audio_int16.tobytes()
            return self.vad.is_speech(frame_bytes, SAMPLE_RATE)
        except Exception as e:
            print(f"VAD error: {e}")
            return False

    def load_models(self):
        """Load Whisper pipeline."""
        self.gui_queue.put(("status", "Loading Model... (this may take a moment)"))
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model="Lingalingeswaran/whisper-small-sinhala_v3",
                device=0 if DEVICE == "cuda" else -1,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
            )
            self.gui_queue.put(("status", "Model Loaded. Click 'Start' to begin."))
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.gui_queue.put(("status", "Falling back to base Whisper model..."))
            try:
                self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-small",
                    device=0 if DEVICE == "cuda" else -1
                )
                self.gui_queue.put(("status", "Model Loaded. Click 'Start' to begin."))
                return True
            except Exception as e2:
                print(f"Fatal error loading model: {e2}")
                self.gui_queue.put(("status", "Error: Model failed to load. Please restart."))
                return False

    def transcribe_audio(self, audio_chunk):
        """Transcribe audio chunk to Sinhala text using the pipeline."""
        try:
            clean_audio = nr.reduce_noise(
                y=audio_chunk,
                sr=SAMPLE_RATE,
                stationary=False,
                prop_decrease=0.8
            )
            clean_audio = clean_audio.astype(np.float32)
            if np.abs(clean_audio).max() > 1.0:
                clean_audio = clean_audio / np.abs(clean_audio).max()

            result = self.pipe(
                clean_audio,
                generate_kwargs={
                    "language": "si",
                    "repetition_penalty": 1.1
                },
                return_timestamps=False
            )
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for real-time audio input."""
        if status:
            print(f"Audio status: {status}")

        if self.stop_event.is_set():
            return

        audio_data = indata.copy().flatten().astype(np.float32)
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1).astype(np.float32)
        audio_data = audio_data / 32768.0

        for i in range(0, len(audio_data), FRAME_SIZE):
            if i + FRAME_SIZE > len(audio_data):
                break
            frame = audio_data[i:i + FRAME_SIZE]
            if len(frame) != FRAME_SIZE:
                continue

            has_speech = self.is_speech_detected(frame)

            if has_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                self.audio_buffer.extend(frame)
                self.silence_counter = 0
            else:
                if self.is_speaking:
                    self.silence_counter += len(frame)
                    self.audio_buffer.extend(frame)
                    if self.silence_counter >= MAX_SILENCE_FRAMES:
                        if len(self.audio_buffer) >= MIN_SPEECH_FRAMES:
                            speech_audio = np.array(self.audio_buffer, dtype=np.float32)
                            self.transcription_queue.put(speech_audio.copy())
                        self.audio_buffer = []
                        self.silence_counter = 0
                        self.is_speaking = False

            if self.is_speaking and len(self.audio_buffer) >= MAX_SPEECH_FRAMES:
                speech_audio = np.array(self.audio_buffer, dtype=np.float32)
                self.transcription_queue.put(speech_audio.copy())
                self.audio_buffer = list(self.audio_buffer[-OVERLAP_FRAMES:])
                self.silence_counter = 0

    def transcription_worker(self):
        """Worker thread for transcription to avoid blocking."""
        if self.pipe is None:
            if not self.load_models():
                return  # Exit thread if model fails to load

        while not self.stop_event.is_set():
            try:
                audio = self.transcription_queue.get(timeout=1)
                if audio is not None and len(audio) > 0:
                    self.gui_queue.put(("status", "Transcribing..."))
                    text = self.transcribe_audio(audio)

                    is_repetitive = False
                    if text and len(text) > 10:
                        first_char = text[0]
                        if text.count(first_char) / len(text) > 0.8:
                            is_repetitive = True
                            print(f"Filtered repetitive output: {text}")

                    if text and not is_repetitive:
                        formatted_text = f"{text} "
                        self.gui_queue.put(("text", formatted_text))

                        try:
                            timestamp = time.strftime("%H:%M:%S")
                            with open('sinhala_transcript.txt', 'a', encoding='utf-8') as f:
                                f.write(f"[{timestamp}] {text}\n")
                        except Exception as e:
                            print(f"Error saving transcript: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")