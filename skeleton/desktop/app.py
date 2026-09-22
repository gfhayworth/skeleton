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
        use_vad: bool = True,
        vad_silence_duration_s: float = 0.45,
        vad_max_recording_s: float = 10.0,
        streaming_playback: bool = True,
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
        self.use_vad = use_vad
        self.vad_silence_duration_s = vad_silence_duration_s
        self.vad_max_recording_s = vad_max_recording_s
        self.streaming_playback = streaming_playback

    def stop(self) -> None:
        """Signals the worker loop to stop after the current step."""
        self.stop_event.set()

    def run(self) -> None:
        """Continuous listen-think-speak loop while active."""
        self.ui_queue.put(("STATUS", "Starting listener..."))

        while not self.stop_event.is_set():
            try:
                # 1. Record user speech from microphone (VAD-streamed or fixed duration)
                self.ui_queue.put(("STATUS", "🎙️ Listening... Speak now!"))
                if self.use_vad:
                    raw_wav = self.recorder.record_with_vad(
                        device_index=self.device_index,
                        silence_duration_s=self.vad_silence_duration_s,
                        max_recording_s=self.vad_max_recording_s,
                    )
                else:
                    raw_wav = self.recorder.record(
                        duration_seconds=self.record_seconds,
                        device_index=self.device_index,
                    )

                if self.stop_event.is_set():
                    break

                # 2. Process audio: choose between ultra-low-latency streaming SSE or batch /audio_in
                self.ui_queue.put(("STATUS", "⚡ Thinking (transcribing & evaluating)..."))
                files = {"file": ("speech.wav", raw_wav, "audio/wav")}

                if self.streaming_playback:
                    self._process_turn_streaming(files)
                else:
                    self._process_turn_batch(files)

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

    def _process_turn_streaming(self, files: dict) -> None:
        """Streams synthesized speech chunks as they are generated, achieving sub-second voice TTFA."""
        user_transcript = ""
        full_text_parts = []
        total_mouth_frames = 0
        total_latency_ms = 0.0
        time_to_first_audio_ms = None

        try:
            with self.client.stream("POST", f"{self.base_url}/audio_in_stream", files=files, timeout=60.0) as resp:
                if resp.status_code != 200:
                    self.ui_queue.put(("ERROR", f"Server error ({resp.status_code})"))
                    time.sleep(1.0)
                    return

                event_type = None
                for line in resp.iter_lines():
                    if self.stop_event.is_set():
                        break

                    if not line:
                        event_type = None
                        continue

                    if line.startswith("event:"):
                        event_type = line[len("event:"):].strip()
                        continue

                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        try:
                            payload = json.loads(data_str)
                        except Exception:
                            continue

                        if event_type == "transcript":
                            user_transcript = payload.get("transcript", "").strip()
                            self.ui_queue.put(("STATUS", f"Transcribed: \"{user_transcript}\" - generating voice..."))

                        elif event_type == "chunk":
                            chunk_text = payload.get("text", "")
                            chunk_b64 = payload.get("audio_base64", "")
                            chunk_latency_ms = payload.get("latency_ms", 0.0)
                            mouth_frames = payload.get("mouth_frames", [])

                            if time_to_first_audio_ms is None:
                                time_to_first_audio_ms = chunk_latency_ms

                            full_text_parts.append(chunk_text)
                            total_mouth_frames += len(mouth_frames)

                            if chunk_b64 and not self.stop_event.is_set():
                                self.ui_queue.put(("STATUS", f"💀 Speaking: \"{chunk_text}\""))
                                wav_bytes = base64.b64decode(chunk_b64)
                                self._play_audio_blocking(wav_bytes)

                        elif event_type == "done":
                            total_latency_ms = payload.get("total_latency_ms", 0.0)

            skeleton_full_text = " ".join(full_text_parts).strip()
            self.ui_queue.put(
                (
                    "TURN",
                    {
                        "user": user_transcript,
                        "skeleton": skeleton_full_text,
                        "latency_ms": time_to_first_audio_ms or total_latency_ms,
                        "mouth_frames": total_mouth_frames,
                    },
                )
            )
            # Acoustic echo silence cooldown
            time.sleep(0.3)

        except Exception as exc:
            if not self.stop_event.is_set():
                self.ui_queue.put(("ERROR", f"Streaming playback error: {exc}"))

    def _process_turn_batch(self, files: dict) -> None:
        """Standard batch audio processing."""
        resp = self.client.post(f"{self.base_url}/audio_in", files=files)
        if self.stop_event.is_set():
            return

        if resp.status_code != 200:
            err_detail = resp.text
            try:
                err_detail = resp.json().get("detail", resp.text)
            except Exception:
                pass
            self.ui_queue.put(("ERROR", f"Server error ({resp.status_code}): {err_detail}"))
            time.sleep(1.0)
            return

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
            time.sleep(0.3)

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

            # Soundboard Quick Trigger Buttons Frame
            soundboard_frame = tk.Frame(self.root, bg="#1e1e2e")
            soundboard_frame.pack(fill=tk.X, padx=16, pady=(0, 8))

            tk.Label(
                soundboard_frame,
                text="Instant Sounds:",
                font=("Segoe UI", 9, "bold"),
                fg="#a6adc8",
                bg="#1e1e2e",
            ).pack(side=tk.LEFT, padx=(0, 6))

            for s_id, label, color in [
                ("laugh_evil_cackle", "💀 Cackle", "#cba6f7"),
                ("filler_hmm_thinking", "🤔 Ponder", "#89dceb"),
                ("greeting_nice_costume", "👋 Greet", "#a6e3a1"),
                ("confused_mumble", "👂 What?", "#f9e2af"),
            ]:
                btn = tk.Button(
                    soundboard_frame,
                    text=label,
                    font=("Segoe UI", 8, "bold"),
                    bg=color,
                    fg="#1e1e2e",
                    relief=tk.FLAT,
                    padx=6,
                    pady=2,
                    cursor="hand2",
                    command=lambda sid=s_id: self._trigger_sound(sid),
                )
                btn.pack(side=tk.LEFT, padx=3)

        def _trigger_sound(self, sound_id: str):
            """Instantly triggers pre-recorded sound over HTTP from the server."""
            target_url = self.url_entry.get().strip().rstrip("/")
            threading.Thread(
                target=self._play_sound_request,
                args=(target_url, sound_id),
                daemon=True,
            ).start()

        def _play_sound_request(self, target_url: str, sound_id: str):
            try:
                resp = httpx.post(f"{target_url}/play_sound/{sound_id}")
                if resp.status_code == 200:
                    data = resp.json()
                    b64 = data.get("audio_base64", "")
                    if b64:
                        wav_bytes = base64.b64decode(b64)
                        self.ui_queue.put(("STATUS", f"Playing pre-recorded sound: {sound_id}"))
                        self.ui_queue.put(
                            (
                                "TURN",
                                {
                                    "user": f"[Instant Trigger: {sound_id}]",
                                    "skeleton": data.get("text", ""),
                                    "latency_ms": 0.0,
                                    "mouth_frames": len(data.get("mouth_frames", [])),
                                },
                            )
                        )
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
                else:
                    self.ui_queue.put(("ERROR", f"Sound trigger failed ({resp.status_code}): {resp.text}"))
            except Exception as e:
                self.ui_queue.put(("ERROR", f"Sound playback error: {e}"))

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
                    use_vad=True,
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
