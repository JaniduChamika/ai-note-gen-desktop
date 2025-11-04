# main.py
# The main entry point for the application.

import tkinter as tk
from gui import TranscriptionApp

if __name__ == "__main__":
    print("=" * 60)
    print("Launching Real-time Sinhala Transcription GUI")
    print("=" * 60)

    # Set up the main application window
    root = tk.Tk()

    # Create an instance of our application
    app = TranscriptionApp(root)

    # Start the GUI event loop
    root.mainloop()

    print("\n✓ Transcription session ended.")