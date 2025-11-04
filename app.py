import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline
import webrtcvad
import noisereduce as nr
import queue
import threading
import time
from collections import deque
# --- NEW IMPORTS ---
import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog
# ---------------------

# Configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHUNK_DURATION = 0.03  # 30ms for VAD frames
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
BUFFER_DURATION = 2.0  # seconds to buffer before transcribing
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
VAD_MODE = 2  # 0-3, aggressiveness of VAD
FRAME_DURATION_MS = 30  # WebRTC VAD frame size in ms (10, 20, or 30)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Speech detection parameters
MIN_SPEECH_DURATION = 0.5  # Minimum speech duration to transcribe (seconds)
MIN_SPEECH_FRAMES = int(MIN_SPEECH_DURATION * SAMPLE_RATE)
MAX_SILENCE_DURATION = 0.6  # Max silence before considering speech ended (seconds)
MAX_SILENCE_FRAMES = int(MAX_SILENCE_DURATION * SAMPLE_RATE)

# new update ---------------------------------------

# Concurrency Tuning (How long to speak before forcing a transcript)
MAX_SPEECH_DURATION = 5.0  # Transcribe every 5 seconds, even without a pause
MAX_SPEECH_FRAMES = int(MAX_SPEECH_DURATION * SAMPLE_RATE)
# Overlap (To prevent cutting words in half between concurrent chunks)
OVERLAP_DURATION = 1.0  # Keep 1 second of the previous audio
OVERLAP_FRAMES = int(OVERLAP_DURATION * SAMPLE_RATE)
# new update ---------------------------------------


# Global variables
audio_buffer = []  # Current speech segment buffer
silence_counter = 0  # Count silence frames
is_speaking = False  # Track if currently in speech segment
transcription_queue = queue.Queue()
stop_event = threading.Event()
vad = webrtcvad.Vad(VAD_MODE)
pipe = None  # Will be initialized in main

# --- NEW GUI QUEUE ---
gui_queue = queue.Queue()


# -----------------------


def is_speech_detected(chunk):
    """Detect speech in a chunk using WebRTC VAD."""
    if len(chunk) != FRAME_SIZE:
        return False

    try:
        # Convert to 16-bit PCM (WebRTC expects int16 bytes)
        # Ensure audio is in proper range
        audio_int16 = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
        frame_bytes = audio_int16.tobytes()
        return vad.is_speech(frame_bytes, SAMPLE_RATE)
    except Exception as e:
        print(f"VAD error: {e}")
        return False


def load_models():
    """Load Whisper pipeline."""
    print("Loading Whisper model for Sinhala...")
    try:
        # Use the specified fine-tuned Sinhala Whisper model
        model_pipe = pipeline(
            "automatic-speech-recognition",
            model="Lingalingeswaran/whisper-small-sinhala_v3",
            device=0 if DEVICE == "cuda" else -1,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        print(f"Model loaded successfully on {DEVICE}.")
        return model_pipe
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to base Whisper model...")
        model_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=0 if DEVICE == "cuda" else -1
        )
        return model_pipe


def audio_callback(indata, frames, time_info, status):
    """Callback for real-time audio input with concurrent transcription."""
    global audio_buffer, silence_counter, is_speaking

    if status:
        print(f"Audio status: {status}")

    # Convert to float32 and handle stereo
    audio_data = indata.copy().flatten().astype(np.float32)
    if indata.shape[1] > 1:  # Stereo - average channels
        audio_data = np.mean(indata, axis=1).astype(np.float32)

    # Normalize to [-1, 1] from int16 range
    audio_data = audio_data / 32768.0

    # Process in VAD frame sizes
    for i in range(0, len(audio_data), FRAME_SIZE):
        if i + FRAME_SIZE > len(audio_data):
            break

        frame = audio_data[i:i + FRAME_SIZE]

        if len(frame) != FRAME_SIZE:
            continue

        has_speech = is_speech_detected(frame)

        if has_speech:
            # --- Speech Detected ---
            if not is_speaking:
                is_speaking = True
                print("Speech started...")

            audio_buffer.extend(frame)
            silence_counter = 0
        else:
            # --- Silence Detected ---
            if is_speaking:
                # We were speaking, but now there's a pause
                silence_counter += len(frame)
                audio_buffer.extend(frame)  # Include the silence for context

                # --- TRIGGER 1: End of Speech (Silence) ---
                if silence_counter >= MAX_SILENCE_FRAMES:
                    if len(audio_buffer) >= MIN_SPEECH_FRAMES:
                        speech_audio = np.array(audio_buffer, dtype=np.float32)
                        transcription_queue.put(speech_audio.copy())
                        print(f"Speech ended (silence). Queued {len(audio_buffer) / SAMPLE_RATE:.2f}s")

                    # Reset everything
                    audio_buffer = []
                    silence_counter = 0
                    is_speaking = False

        # --- TRIGGER 2: Concurrent Transcription (Max Time) ---
        # This check runs every frame, even during speech
        # It ensures we transcribe if the user speaks for too long without pausing
        if is_speaking and len(audio_buffer) >= MAX_SPEECH_FRAMES:
            print(f"Concurrent chunk (max time). Queued {len(audio_buffer) / SAMPLE_RATE:.2f}s")
            speech_audio = np.array(audio_buffer, dtype=np.float32)
            transcription_queue.put(speech_audio.copy())

            # Reset, but KEEP the overlap
            audio_buffer = list(audio_buffer[-OVERLAP_FRAMES:])
            silence_counter = 0  # We are still speaking
            # is_speaking remains True


def transcribe_audio(audio_chunk):
    """Transcribe audio chunk to Sinhala text using the pipeline."""
    try:
        # Apply noise reduction
        clean_audio = nr.reduce_noise(
            y=audio_chunk,
            sr=SAMPLE_RATE,
            stationary=True,
            prop_decrease=0.9
        )

        # Ensure audio is float32 in [-1,1]
        clean_audio = clean_audio.astype(np.float32)
        if np.abs(clean_audio).max() > 1.0:
            clean_audio = clean_audio / np.abs(clean_audio).max()

        # Transcribe using pipeline
        result = pipe(
            clean_audio,
            generate_kwargs={"language": "si", "task": "transcribe", "repetition_penalty": 1.1},
            return_timestamps=False
        )

        text = result["text"].strip()
        return text
    except Exception as e:
        print(f"Transcription error: {e}")
        return None


def transcription_worker():
    """Worker thread for transcription to avoid blocking."""
    global pipe

    # 1. Load model inside the thread to avoid freezing the GUI
    if pipe is None:
        # Send a "status" message
        gui_queue.put(("status", "Loading Model... (this may take a moment)"))
        pipe = load_models()
        if pipe:
            # Send a "status" message
            gui_queue.put(("status", "Model Loaded. Click 'Start' to begin."))
        else:
            # Send a "status" message
            gui_queue.put(("status", "Error: Model failed to load. Please restart."))
            return

    while not stop_event.is_set():
        try:
            audio = transcription_queue.get(timeout=1)
            if audio is not None and len(audio) > 0:
                # Send a "status" message
                gui_queue.put(("status", "Transcribing..."))
                text = transcribe_audio(audio)

                is_repetitive = False
                if text and len(text) > 10:
                    first_char = text[0]
                    if text.count(first_char) / len(text) > 0.8:
                        is_repetitive = True
                        print(f"Filtered repetitive output: {text}")

                if text and not is_repetitive:

                    # --- THIS IS THE KEY CHANGE ---
                    # Instead of a formatted line, send just the text + a space
                    formatted_text = f"{text} "
                    # Send a "text" message
                    gui_queue.put(("text", formatted_text))
                    # ------------------------------

                    # Optional: Save to file (no change here)
                    try:
                        timestamp = time.strftime("%H:%M:%S")  # Timestamp only for file
                        with open('sinhala_transcript.txt', 'a', encoding='utf-8') as f:
                            f.write(f"[{timestamp}] {text}\n")
                    except Exception as e:
                        print(f"Error saving transcript: {e}")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker error: {e}")

# -------------------------------------------------


# --- NEW: GUI Application Class ---

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Sinhala Transcription")
        self.root.geometry("600x400")

        self.is_running = False
        self.stream = None
        self.worker_thread = None

        # --- Create GUI Elements ---

        # Title Label
        self.title_label = tk.Label(root, text="Sinhala Real-time Transcription", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)

        # Start Button
        self.start_button = tk.Button(root, text="Start", command=self.start_transcription, font=("Arial", 12),
                                      bg="#4CAF50", fg="white", width=15)
        self.start_button.pack(pady=5)

        # Stop Button
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_transcription, font=("Arial", 12),
                                     bg="#F44336", fg="white", width=15, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        # --- NEW: Save Button ---
        self.save_button = tk.Button(root, text="Save Transcript", command=self.save_transcript, font=("Arial", 12),
                                     bg="#008CBA", fg="white", width=15, state=tk.DISABLED)
        self.save_button.pack(pady=(0, 10))  # Add a little space below

        # --- NEW: Status Label ---
        self.status_label = tk.Label(root, text="Loading model... please wait.", font=("Arial", 10, "italic"),
                                     fg="gray")
        self.status_label.pack(pady=(0, 5))

        # Live Transcription Area
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=15, font=("Arial", 10))
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # --- App Logic ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.check_gui_queue()  # Start the GUI queue checker

        # Pre-load the model in the background
        self.start_worker_thread()

    def check_gui_queue(self):
        """Check the queue for new text and update the GUI."""
        try:
            message = gui_queue.get_nowait()

            # Check if the message is our new tuple format
            if isinstance(message, tuple) and len(message) == 2:
                msg_type, content = message

                if msg_type == "status":
                    # Update the status label
                    self.status_label.config(text=content, fg="blue")
                elif msg_type == "text":
                    # Append text to the text area
                    self.text_area.insert(tk.END, content)
                    self.text_area.see(tk.END)  # Auto-scroll

                    # Once we get text, set status back to "Listening"
                    self.status_label.config(text="Listening...", fg="green")

        except queue.Empty:
            pass
        # Check again after 100ms
        self.root.after(100, self.check_gui_queue)

    def start_worker_thread(self):
        """Starts the transcription worker thread."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=transcription_worker, daemon=True)
            self.worker_thread.start()

    def start_transcription(self):
        if self.is_running:
            return

        if pipe is None:
            self.status_label.config(text="Model is still loading, please wait...", fg="orange")
            self.start_worker_thread()
            return

        print("Starting audio stream...")
        self.is_running = True

        # --- NEW: Clear the text area on start ---
        self.text_area.delete(1.0, tk.END)
        # ----------------------------------------

        global audio_buffer, silence_counter, is_speaking
        audio_buffer = []
        silence_counter = 0
        is_speaking = False

        # Update status label
        self.status_label.config(text="Listening... Speak in Sinhala.", fg="green")

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='int16',
                blocksize=CHUNK_SIZE,
                callback=audio_callback
            )
            self.stream.start()

            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)

        except Exception as e:
            # Update status label with error
            self.status_label.config(text=f"Error starting audio stream: {e}", fg="red")
            self.is_running = False

    def stop_transcription(self):
        if not self.is_running:
            return

        print("Stopping audio stream...")
        stop_event.set()

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        stop_event.clear()
        self.is_running = False

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)

        # Update status label
        self.status_label.config(text="Stopped. Click 'Start' to begin.", fg="gray")
    def on_closing(self):
        """Handle window close event."""
        if self.is_running:
            self.stop_transcription()
        stop_event.set()  # Tell worker thread to exit
        if self.worker_thread:
            self.worker_thread.join(timeout=1)  # Wait for worker
        self.root.destroy()

    def save_transcript(self):
        """Opens a 'Save As' dialog to save the transcript."""
        print("Opening save dialog...")

        # Get the full text from the text area
        # 1.0 means "line 1, character 0"
        # tk.END means "to the very end"
        # -1c removes the automatic newline tkinter adds at the end
        transcript_text = self.text_area.get("1.0", tk.END + "-1c")

        if not transcript_text:
            self.status_label.config(text="Nothing to save.", fg="orange")
            return

        try:
            # Open the 'Save As' dialog
            file_path = filedialog.asksaveasfilename(
                initialfile="sinhala_transcript.txt",
                defaultextension=".txt",
                filetypes=[
                    ("Text Files", "*.txt"),
                    ("All Files", "*.*")
                ]
            )

            # If the user cancels, file_path will be empty
            if not file_path:
                print("Save cancelled by user.")
                self.status_label.config(text="Save cancelled.", fg="gray")
                return

            # Write the text to the chosen file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)

            print(f"Transcript saved to: {file_path}")
            self.status_label.config(text=f"Transcript saved to {file_path}", fg="green")

        except Exception as e:
            print(f"Error saving file: {e}")
            self.status_label.config(text=f"Error saving file: {e}", fg="red")


if __name__ == "__main__":
    print("=" * 60)
    print("Launching Real-time Sinhala Transcription GUI")
    print("=" * 60)

    # Set up the main application window
    root = tk.Tk()
    app = TranscriptionApp(root)

    # Start the GUI event loop
    root.mainloop()

    print("\n✓ Transcription session ended.")
