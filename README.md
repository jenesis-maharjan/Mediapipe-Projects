# 👆 MediaPipe Projects

Two interactive hand-tracking applications using Google MediaPipe. Play music and count fingers using just your webcam - no special hardware needed!

---

## 🎯 Projects Overview

### 1️⃣ Finger Counter
**Description:** Counts how many fingers you're holding up (0-5) in real-time. Green dots show extended fingers, red dots show folded fingers.

**Features:**
- Real-time finger counting (0-5 fingers)
- Visual feedback with colored dots
- Emoji display (✌️ for 2, 🖐️ for 5)
- Screenshot capture

**Use cases:** Learning hand tracking basics, gesture recognition, sign language practice

---

### 2️⃣ Virtual Piano
**Description:** Play piano by moving your index finger over virtual keys. Eight musical notes, two playing modes, supports both hands!

**Features:**
- 8 musical notes (Do, Re, Mi, Fa, So, La, Ti, Do²)
- Two playing modes (Zone mode + Pinch mode)
- Both hands supported (play chords or duets)
- Beautiful gradient piano keys with glow effects
- Real-time FPS counter

**Use cases:** Music learning, contactless instrument, fun interactive project

---

## 🎮 Controls

| Project | Key | Action |
|---------|-----|--------|
| **Both** | `Q` | Quit application |
| **Both** | `S` | Take screenshot |
| **Virtual Piano** | `M` | Toggle mode (zone/pinch) |
| **Virtual Piano** | `+` / `=` | Faster response |
| **Virtual Piano** | `-` | Slower response |

---

## 🎵 Playing Modes (Virtual Piano)

| Mode | How it works | Best for |
|------|-------------|----------|
| **ZONE Mode** | Finger enters key zone = plays note | Beginners, casual playing |
| **PINCH Mode** | Finger over key + pinch thumb & index = plays note | Precise control, preventing accidents |

---


## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Main programming language |
| **OpenCV** | Camera capture and image processing |
| **MediaPipe** | Hand landmark detection (21 points) |
| **PyGame** | Audio playback for piano notes |
| **NumPy** | Mathematical operations |

---

## Clone repository
git clone https://github.com/jenesis-maharjan/MediaPipe-Projects.git
