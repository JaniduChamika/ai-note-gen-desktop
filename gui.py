# gui.py
# Contains all Tkinter GUI logic for the application

import tkinter as tk
from tkinter import scrolledtext, filedialog
import threading
import queue
import sounddevice as sd

# Import our new engine
from asr_engine import AsrEngine
# Import constants
from config import *


class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Sinhala Transcription")
        self.root.geometry("600x400")

        self.is_running = False
        self.stream = None
        self.worker_thread = None

        # --- Create Queues and Engine ---
        self.gui_queue = queue.Queue()
        self.engine = AsrEngine(self.gui_queue)  # Create the engine instance

        # --- Create GUI Elements ---
        self.title_label = tk.Label(root, text="Sinhala Real-time Transcription", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)

        self.start_button = tk.Button(root, text="Start", command=self.start_transcription, font=("Arial", 12),
                                      bg="#4CAF50", fg="white", width=15)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_transcription, font=("Arial", 12),
                                     bg="#F44336", fg="white", width=15, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.save_button = tk.Button(root, text="Save Transcript", command=self.save_transcript, font=("Arial", 12),
                                     bg="#008CBA", fg="white", width=15, state=tk.DISABLED)
        self.save_button.pack(pady=(0, 10))

        self.status_label = tk.Label(root, text="Loading model... please wait.", font=("Arial", 10, "italic"),
                                     fg="gray")
        self.status_label.pack(pady=(0, 5))

        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=15, font=("Arial", 10))
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # --- App Logic ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.check_gui_queue()
        self.start_worker_thread()

    def check_gui_queue(self):
        """Check the queue for new messages and update the GUI."""
        try:
            message = self.gui_queue.get_nowait()
            if isinstance(message, tuple) and len(message) == 2:
                msg_type, content = message
                if msg_type == "status":
                    self.status_label.config(text=content, fg="blue" if "Transcribing" in content else "gray")
                elif msg_type == "text":
                    self.text_area.insert(tk.END, content)
                    self.text_area.see(tk.END)
                    self.status_label.config(text="Listening...", fg="green")
        except queue.Empty:
            pass
        self.root.after(100, self.check_gui_queue)

    def start_worker_thread(self):
        """Starts the transcription worker thread from the engine."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=self.engine.transcription_worker,  # Run the engine's worker
                daemon=True
            )
            self.worker_thread.start()

    def start_transcription(self):
        if self.is_running:
            return
        if self.engine.pipe is None:
            self.status_label.config(text="Model is still loading, please wait...", fg="orange")
            self.start_worker_thread()
            return

        print("Starting audio stream...")
        self.is_running = True
        self.engine.reset()  # Reset the engine's state

        self.text_area.delete(1.0, tk.END)
        self.status_label.config(text="Listening... Speak in Sinhala.", fg="green")

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='int16',
                blocksize=CHUNK_SIZE,
                callback=self.engine.audio_callback  # Use the engine's callback
            )
            self.stream.start()

            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)

        except Exception as e:
            self.status_label.config(text=f"Error starting audio stream: {e}", fg="red")
            self.is_running = False

    def stop_transcription(self):
        if not self.is_running:
            return

        print("Stopping audio stream...")
        self.engine.stop()  # Tell the engine to stop

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self.status_label.config(text="Stopped. Click 'Start' to begin.", fg="gray")

    def save_transcript(self):
        """Opens a 'Save As' dialog to save the transcript."""
        print("Opening save dialog...")
        transcript_text = self.text_area.get("1.0", tk.END + "-1c")

        if not transcript_text:
            self.status_label.config(text="Nothing to save.", fg="orange")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                initialfile="sinhala_transcript.txt",
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if not file_path:
                print("Save cancelled by user.")
                self.status_label.config(text="Save cancelled.", fg="gray")
                return

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)

            self.status_label.config(text=f"Transcript saved to {file_path}", fg="green")
        except Exception as e:
            print(f"Error saving file: {e}")
            self.status_label.config(text=f"Error saving file: {e}", fg="red")

    def on_closing(self):
        """Handle window close event."""
        if self.is_running:
            self.stop_transcription()
        self.engine.stop()  # Ensure engine threads stop
        if self.worker_thread:
            self.worker_thread.join(timeout=1)
        self.root.destroy()