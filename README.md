# 💀 Animatronic Skeleton Audio & Dialogue Subsystem

A low-latency, real-time voice conversation and physical jaw-synchronization system for an animatronic Halloween skeleton. 

The system couples Google Gemini for conversational banter, OpenAI Whisper for speech-to-text, OpenAI Onyx for speech synthesis, real-time 30Hz servo jaw tracking, and an instant pre-recorded soundboard.

---

## 📋 System Requirements & Prerequisites

- **Operating System:** Windows 10/11, macOS, or Linux
- **Python Version:** Python 3.10 to 3.13 (tested on Python 3.13)
- **Microphone & Speakers:** Working audio input and output devices
- **API Keys:**
  - **Google Gemini API Key:** Powers the conversational dialogue orchestrator (`gemini-2.5-flash`).
  - **OpenAI API Key:** Powers Whisper speech recognition (`whisper-1`) and deep male voice synthesis (`tts-1`, voice `onyx`).

---

## 🛠️ Environment Setup & Virtual Environment

### 1. Clone or Open the Repository
```powershell
git clone https://github.com/gfhayworth/skeleton.git
cd skeleton
```

### 2. Create and Activate a Virtual Environment (`.venv`)

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted on Windows, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` first).*

**On Linux / macOS (Bash / Zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
With your virtual environment activated, install all dependencies from `requirements.txt`:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Key packages installed:
- `fastapi`, `uvicorn`, `websockets`, `sse-starlette` (Backend server & real-time streaming)
- `sounddevice`, `numpy`, `webrtcvad-wheels` (Microphone capture & Voice Activity Detection)
- `httpx`, `pydantic`, `pydantic-settings`, `python-dotenv` (Network clients & configuration)
- `pytest`, `pytest-asyncio` (Automated testing suite)
- `ipykernel`, `notebook`, `ipywidgets` (Interactive Jupyter demo)

### 4. Configure API Keys (`.env`)
Create a `.env` file in the root directory of the project:
```ini
# Google Gemini API key (for low-latency dialogue generation)
GEMINI_API_KEY=your-gemini-api-key-here

# OpenAI API key (for Whisper STT and Onyx TTS audio synthesis)
OPENAI_API_KEY=your-openai-api-key-here
```

---

## 🚀 Quickstart: Running the System

The system operates via a client-server architecture. You run the backend server in one terminal and the desktop listener in another.

### Step 1: Start the Backend API Server
In **Terminal 1**, run Uvicorn to host the audio processing, Whisper, Gemini, and WebSocket endpoints:
```powershell
.\.venv\Scripts\python.exe -m uvicorn skeleton.server.app:app --port 8000
```
*The server will be ready on `http://127.0.0.1:8000`.*

### Step 2: Launch the Desktop Listener Application
In **Terminal 2**, launch the Tkinter desktop GUI:
```powershell
.\.venv\Scripts\python.exe -m skeleton.desktop
```

---

## 🖥️ Using the Desktop Listener Application

When the desktop window appears:

1. **Transport Mode:** Choose your preferred network transport:
   - **`WebSocket (Fastest)` (Recommended):** Full-duplex persistent socket (`/ws/audio`) for minimal audio turnaround latency.
   - **`SSE Streaming`:** Server-Sent Events (`/audio_in_stream`) for chunked sentence-by-sentence streaming.
   - **`Batch HTTP`:** Standard request/response (`/audio_in`).
2. **Auto-Greet Checkbox:**
   - When checked (default), the skeleton **instantly responds to your very first words** with a randomly selected pre-recorded greeting (< 20ms turnaround), without waiting for speech-to-text or LLM processing.
   - Turn 2 and onward automatically transition to full AI dialogue.
3. **Start Listening:** Click **`🔴 LISTENING: OFF (Click to Start)`** to enable the microphone.
   - The button turns green: **`🟢 LISTENING: ON`**.
   - Speak into your microphone. Voice Activity Detection (VAD) automatically senses when you finish speaking.
4. **Instant Soundboard:** Click any of the bottom buttons for zero-latency canned reactions:
   - **💀 Cackle:** Plays evil skeletal laughter.
   - **🤔 Ponder:** Plays a thinking filler sound (*"Hmm, let me ponder your existence..."*).
   - **👋 Greet:** Plays a random greeting insult.
   - **👂 What?:** Plays a confused hearing snark.

---

## 📓 Interactive Jupyter Notebook Demo

To test and explore the dialogue subsystem interactively in a Jupyter notebook:
```powershell
.\.venv\Scripts\python.exe -m notebook notebooks/dialogue_demo.ipynb
```
The notebook demonstrates:
- Prompt isolation and history bounded pruning
- Live Gemini streaming with clause-based sentence chunking
- Deep male **Onyx** TTS synthesis and audio playback
- 30Hz jaw servo trajectory graphs

---

## 🔊 Re-Baking Pre-Recorded Sounds

The project includes 23 pre-recorded audio clips in `assets/sounds/` with pre-computed 30Hz servo mouth-sync trajectories.

If you edit the script definitions in `skeleton/audio/bake_sounds.py` or want to regenerate all audio assets using your OpenAI API key:
```powershell
.\.venv\Scripts\python.exe -m skeleton.audio.bake_sounds
```
*To generate offline synthetic tones without API calls, pass `--mock`.*

---

## 🧪 Running the Test Suite

Run the full automated test suite (66 tests covering all subsystems):
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
Run individual test modules:
```powershell
# Desktop listener and worker tests
.\.venv\Scripts\python.exe -m pytest tests/test_desktop.py -v

# Server API & WebSocket streaming tests
.\.venv\Scripts\python.exe -m pytest tests/test_server.py tests/test_websocket.py -v

# Mouth sync and audio pipeline tests
.\.venv\Scripts\python.exe -m pytest tests/test_mouth_sync.py tests/test_sounds.py -v
```

---

## 📁 Project Architecture

```
skeleton/
├── assets/
│   └── sounds/              # 23 baked WAV audio files & 30Hz jaw trajectory JSONs
├── notebooks/
│   └── dialogue_demo.ipynb  # Interactive Jupyter demo notebook
├── skeleton/
│   ├── audio/               # Whisper STT, Onyx TTS, VAD Recorder, Mouth-Sync
│   │   ├── bake_sounds.py   # Sound asset baking utility
│   │   ├── config.py        # Subsystem audio configuration
│   │   ├── mouth_sync.py    # 30Hz RMS jaw servo trajectory processor
│   │   ├── pipeline.py      # Master audio pipeline orchestrator
│   │   ├── recorder.py      # WebRTC VAD microphone capture
│   │   ├── sounds.py        # SoundBank cache manager
│   │   ├── stt.py           # Whisper STT client
│   │   └── tts.py           # OpenAI TTS client (Onyx voice)
│   ├── desktop/             # Desktop listener application
│   │   ├── __init__.py      # PEP 562 lazy attribute loading
│   │   ├── __main__.py      # Module entrypoint (python -m skeleton.desktop)
│   │   └── app.py           # Tkinter GUI and AudioWorker thread
│   ├── dialogue/            # Dialogue engine (Gemini & persona)
│   │   ├── chunker.py       # Sentence & clause streaming chunker
│   │   ├── llm_client.py    # Gemini & Mock LLM clients
│   │   ├── orchestrator.py  # Conversational turn manager
│   │   ├── prompt_builder.py# System prompt & history pruner
│   │   └── sanitizer.py     # Stage direction & emoji stripper
│   └── server/              # FastAPI backend server
│       ├── app.py           # REST endpoints (/audio_in, /audio_out, /sounds)
│       └── ws.py            # Duplex WebSocket streaming (/ws/audio)
├── tests/                   # 66 comprehensive automated unit & integration tests
├── requirements.txt         # Project dependencies
└── README.md                # This documentation
```
