"""Desktop listener application with ON/OFF toggle for the animatronic skeleton."""

import base64
import io
import os
import queue
import sys
import threading
import time
import wave
from typing import Optional
import httpx
import numpy as np
import sounddevice as sd

from skeleton.audio.recorder import AudioRecorder, BaseAudioRecorder, MockAudioRecorder


class AudioWorker(threading.Thread):
    """Background worker thread that records microphone audio, sends it to /audio_in,

    and vocalizes the skeleton's response through speakers with acoustic echo protection.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        record_seconds: float = 3.5,
        device_index: Optional[int] = None,
        recorder: Optional[BaseAudioRecorder] = None,
        ui_queue: Optional[queue.Queue] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        super().__init__(daemon=True)
        self.base_url = base_url.rstrip("/")
        self.record_seconds = record_seconds
        self.device_index = device_index
        self.recorder = recorder or AudioRecorder()
        self.ui_queue = ui_queue or queue.Queue()
        self.client = http_client or httpx.Client(timeout=30.0)
        self._owns_client = http_client is None
        self.stop_event = threading.Event()

    def stop(self) -> None:
        """Signals the worker loop to stop after the current step."""
        self.stop_event.set()

    def run(self) -> None:
        """Continuous listen-think-speak loop while active."""
        self.ui_queue.put(("STATUS", "Starting listener..."))

        while not self.stop_event.is_set():
            try:
                # 1. Record user speech from microphone
                self.ui_queue.put(("STATUS", "🎙️ Listening... Speak now!"))
                raw_wav = self.recorder.record(
                    duration_seconds=self.record_seconds,
                    device_index=self.device_index,
                )

                if self.stop_event.is_set():
                    break

                # 2. Upload to FastAPI /audio_in
                self.ui_queue.put(("STATUS", "⚡ Thinking (transcribing & evaluating)..."))
                files = {"file": ("speech.wav", raw_wav, "audio/wav")}
                resp = self.client.post(f"{self.base_url}/audio_in", files=files)

                if self.stop_event.is_set():
                    break

                if resp.status_code != 200:
                    err_detail = resp.text
                    try:
                        err_detail = resp.json().get("detail", resp.text)
                    except Exception:
                        pass
                    self.ui_queue.put(("ERROR", f"Server error ({resp.status_code}): {err_detail}"))
                    time.sleep(1.0)
                    continue

                data = resp.json()
                user_transcript = data.get("user_transcript", "").strip()
                skeleton_response = data.get("skeleton_response", "").strip()
                audio_b64 = data.get("audio_base64", "")
                latency_ms = data.get("total_latency_ms", 0.0)
                mouth_frames_count = len(data.get("mouth_frames", []))

                # Post conversation turn to UI
                self.ui_queue.put(
                    (
                        "TURN",
                        {
                            "user": user_transcript,
                            "skeleton": skeleton_response,
                            "latency_ms": latency_ms,
                            "mouth_frames": mouth_frames_count,
                        },
                    )
                )

                # 3. Vocalize response over speakers (blocking + acoustic echo cooldown)
                if audio_b64 and not self.stop_event.is_set():
                    self.ui_queue.put(("STATUS", "💀 Speaking response..."))
                    wav_bytes = base64.b64decode(audio_b64)
                    self._play_audio_blocking(wav_bytes)

                    # Acoustic echo silence cooldown: Wait 300ms so the microphone
                    # doesn't immediately hear room reflections of the skeleton's voice
                    time.sleep(0.3)

            except Exception as exc:
                if self.stop_event.is_set():
                    break
                self.ui_queue.put(("ERROR", f"Audio loop error: {exc}"))
                time.sleep(1.5)

        self.ui_queue.put(("STATUS", "Idle (Listening OFF)"))
        if self._owns_client:
            try:
                self.client.close()
            except Exception:
                pass

    def _play_audio_blocking(self, wav_bytes: bytes) -> None:
        """Plays PCM WAV audio through local speakers and waits for completion."""
        try:
            with io.BytesIO(wav_bytes) as buf:
                with wave.open(buf, "rb") as wf:
                    channels = wf.getnchannels()
                    rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    if channels > 1:
                        audio_data = audio_data.reshape(-1, channels)

                    sd.play(audio_data, samplerate=rate)
                    sd.wait()
        except Exception as exc:
            self.ui_queue.put(("ERROR", f"Audio playback failed: {exc}"))


# ---------------------------------------------------------------------------
# Tkinter Desktop GUI
# ---------------------------------------------------------------------------


def launch_gui(base_url: str = "http://127.0.0.1:8000") -> None:
    """Launches the Tkinter desktop client window."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    class SkeletonDesktopApp:
        def __init__(self, root: tk.Tk):
            self.root = root
            self.root.title("Skeleton Animatronic Listener")
            self.root.geometry("640x580")
            self.root.minsize(500, 450)

            self.ui_queue: queue.Queue = queue.Queue()
            self.worker: Optional[AudioWorker] = None
            self.is_listening = False
            self.turn_count = 0

            self._create_widgets()
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self.root.after(50, self._process_ui_queue)

        def _create_widgets(self):
            # Styling
            self.root.configure(bg="#1e1e2e")
            style = ttk.Style()
            style.theme_use("clam")

            # Header Frame
            header_frame = tk.Frame(self.root, bg="#1e1e2e")
            header_frame.pack(fill=tk.X, padx=16, pady=(12, 6))

            title_lbl = tk.Label(
                header_frame,
                text="💀 Skeleton Animatronic Audio Controller",
                font=("Segoe UI", 14, "bold"),
                fg="#f8f8f2",
                bg="#1e1e2e",
            )
            title_lbl.pack(side=tk.LEFT)

            # Server URL Config Frame
            url_frame = tk.Frame(self.root, bg="#1e1e2e")
            url_frame.pack(fill=tk.X, padx=16, pady=4)

            tk.Label(url_frame, text="Server API:", font=("Segoe UI", 9), fg="#a6adc8", bg="#1e1e2e").pack(
                side=tk.LEFT, padx=(0, 6)
            )
            self.url_entry = tk.Entry(url_frame, font=("Segoe UI", 9), width=32, bg="#313244", fg="#cdd6f4")
            self.url_entry.insert(0, base_url)
            self.url_entry.pack(side=tk.LEFT)

            # Master ON / OFF Toggle Button
            self.toggle_btn = tk.Button(
                self.root,
                text="🔴 LISTENING: OFF (Click to Start)",
                font=("Segoe UI", 13, "bold"),
                bg="#e78284",
                fg="#ffffff",
                activebackground="#ea999c",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                padx=16,
                pady=10,
                cursor="hand2",
                command=self._toggle_listening,
            )
            self.toggle_btn.pack(fill=tk.X, padx=16, pady=10)

            # Status Banner
            self.status_lbl = tk.Label(
                self.root,
                text="Status: Idle (Click button above to enable microphone)",
                font=("Segoe UI", 10, "italic"),
                fg="#bac2de",
                bg="#181825",
                pady=6,
            )
            self.status_lbl.pack(fill=tk.X, padx=16, pady=2)

            # Dialogue History Text Box
            log_frame = tk.Frame(self.root, bg="#1e1e2e")
            log_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(8, 4))

            tk.Label(
                log_frame,
                text="Conversation Transcript:",
                font=("Segoe UI", 10, "bold"),
                fg="#cdd6f4",
                bg="#1e1e2e",
            ).pack(anchor=tk.W, pady=(0, 4))

            self.text_log = tk.Text(
                log_frame,
                wrap=tk.WORD,
                font=("Consolas", 10),
                bg="#11111b",
                fg="#cdd6f4",
                insertbackground="#f5e0dc",
                relief=tk.FLAT,
                padx=8,
                pady=8,
            )
            self.text_log.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

            scrollbar = tk.Scrollbar(log_frame, command=self.text_log.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.text_log.config(yscrollcommand=scrollbar.set)

            # Tags for text styling
            self.text_log.tag_config("user_tag", foreground="#89b4fa", font=("Consolas", 10, "bold"))
            self.text_log.tag_config("skeleton_tag", foreground="#a6e3a1", font=("Consolas", 10, "bold"))
            self.text_log.tag_config("telemetry_tag", foreground="#6c7086", font=("Consolas", 9, "italic"))
            self.text_log.tag_config("error_tag", foreground="#f38ba8", font=("Consolas", 9, "bold"))

            # Footer / Telemetry Badge
            self.footer_lbl = tk.Label(
                self.root,
                text="Turns: 0 | Latency: -- ms | Mouth Frames: --",
                font=("Segoe UI", 8),
                fg="#6c7086",
                bg="#1e1e2e",
                pady=4,
            )
            self.footer_lbl.pack(fill=tk.X, padx=16, pady=(0, 6))

        def _toggle_listening(self):
            if not self.is_listening:
                # Turn ON
                target_url = self.url_entry.get().strip()
                if not target_url:
                    messagebox.showerror("Error", "Server API URL cannot be empty.")
                    return

                self.is_listening = True
                self.toggle_btn.config(
                    text="🟢 LISTENING: ON (Click to Stop)",
                    bg="#a6e3a1",
                    fg="#1e1e2e",
                    activebackground="#94e2d5",
                )
                self.url_entry.config(state=tk.DISABLED)

                self.worker = AudioWorker(
                    base_url=target_url,
                    record_seconds=3.5,
                    ui_queue=self.ui_queue,
                )
                self.worker.start()
            else:
                # Turn OFF
                self.is_listening = False
                self.toggle_btn.config(
                    text="🔴 LISTENING: OFF (Click to Start)",
                    bg="#e78284",
                    fg="#ffffff",
                    activebackground="#ea999c",
                )
                self.url_entry.config(state=tk.NORMAL)
                self.status_lbl.config(text="Status: Stopping listener...")

                if self.worker is not None:
                    self.worker.stop()
                    self.worker = None

        def _process_ui_queue(self):
            """Marshals events from worker thread strictly onto the Tkinter main thread."""
            try:
                while True:
                    event_type, payload = self.ui_queue.get_nowait()
                    if event_type == "STATUS":
                        self.status_lbl.config(text=f"Status: {payload}")
                    elif event_type == "TURN":
                        self.turn_count += 1
                        user_text = payload.get("user", "")
                        skeleton_text = payload.get("skeleton", "")
                        latency = payload.get("latency_ms", 0.0)
                        frames = payload.get("mouth_frames", 0)

                        self.text_log.insert(tk.END, f"\nTurn #{self.turn_count}\n", "telemetry_tag")
                        self.text_log.insert(tk.END, "You: ", "user_tag")
                        self.text_log.insert(tk.END, f"{user_text}\n")
                        self.text_log.insert(tk.END, "Skeleton: ", "skeleton_tag")
                        self.text_log.insert(tk.END, f"{skeleton_text}\n")
                        self.text_log.see(tk.END)

                        self.footer_lbl.config(
                            text=f"Turns: {self.turn_count} | Latency: {latency:.0f} ms | Mouth Frames: {frames}"
                        )
                    elif event_type == "ERROR":
                        self.text_log.insert(tk.END, f"\n[Error] {payload}\n", "error_tag")
                        self.text_log.see(tk.END)
                        self.status_lbl.config(text=f"Status: {payload}")

                    self.ui_queue.task_done()
            except queue.Empty:
                pass

            # Continue polling every 50ms
            self.root.after(50, self._process_ui_queue)

        def _on_close(self):
            """Clean shutdown of worker thread and audio streams before destroying window."""
            if self.worker is not None:
                self.worker.stop()
                self.worker.join(timeout=1.0)
            self.root.destroy()

    root = tk.Tk()
    app = SkeletonDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
