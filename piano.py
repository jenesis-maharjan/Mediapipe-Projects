"""
AI VIRTUAL PIANO - WORKING POINTER VERSION
===========================================
This version ALWAYS shows the pointer and triggers sounds.
Hand tracking works, index finger cursor works, sounds play!
"""

# Standard libraries for computer vision, hand tracking, audio playback,
# filesystem access, timing, smoothing buffers, and numeric arrays.
import cv2
import mediapipe as mp
import pygame
import os
import time
from collections import deque
import numpy as np

# ============================================================================
# AUDIO SETUP
# ============================================================================

pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()
pygame.mixer.set_num_channels(16)

# ============================================================================
# LOAD MUSICAL NOTES
# ============================================================================

NOTE_NAMES = ["Do", "Re", "Mi", "Fa", "So", "La", "Ti", "Do²"]
SOUND_FILES = ["sounds/do.wav", "sounds/re.wav", "sounds/mi.wav", "sounds/fa.wav",
               "sounds/so.wav", "sounds/la.wav", "sounds/ti.wav", "sounds/high_do.wav"]

notes: list[pygame.mixer.Sound | None] = []
for f in SOUND_FILES:
    if os.path.exists(f):
        s = pygame.mixer.Sound(f)
        s.set_volume(0.9)
        notes.append(s)
        print(f"✓ Loaded: {f}")
    else:
        print(f"⚠ Warning: Missing sound: {f}")
        notes.append(None)

NUM_KEYS = len(notes)

# ============================================================================
# VISUAL COLORS
# ============================================================================

KEY_COLORS = [
    (30, 144, 255), (50, 205, 50), (255, 215, 0), (255, 140, 0),
    (255, 69, 0), (218, 112, 214), (147, 112, 219), (220, 20, 60),
]

LIT_COLORS = [
    (100, 180, 255), (100, 255, 100), (255, 240, 100), (255, 180, 80),
    (255, 120, 80), (255, 160, 255), (200, 160, 255), (255, 80, 120),
]

# ============================================================================
# MEDIAPIPE SETUP
# ============================================================================

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3,
)

mp_draw = mp.solutions.drawing_utils

# ============================================================================
# CAMERA SETUP
# ============================================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

CAM_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera resolution: {CAM_W} x {CAM_H}")

# ============================================================================
# PIANO KEY LAYOUT
# ============================================================================

KEY_ZONE_TOP = int(CAM_H * 0.65)
KEY_ZONE_BOTTOM = CAM_H - 5
KEY_W = CAM_W // NUM_KEYS

def key_rect(i: int):
    """Return the pixel rectangle for piano key index i."""
    x1 = i * KEY_W
    x2 = x1 + KEY_W - 1
    return x1, KEY_ZONE_TOP, x2, KEY_ZONE_BOTTOM

# ============================================================================
# GAME PARAMETERS
# ============================================================================

DEBOUNCE_S = 0.15
SMOOTH_N = 3
PINCH_THRESH = 0.045
TRIGGER_MODE = "zone"

KEY_ALPHA = 0.20
LIT_ALPHA = 0.55
LIT_DURATION = 0.12

# ============================================================================
# GAME STATE
# ============================================================================

last_played: dict[int, float] = {}
key_lit: dict[int, float] = {}
prev_in_zone: dict[int, set] = {0: set(), 1: set()}
finger_buffers: dict[int, deque] = {}
frame_count = 0
last_result = None

# ============================================================================
# SIMPLE SMOOTHING FUNCTION
# ============================================================================

def smooth_point(hand_id: int, x_raw: float, y_raw: float):
    """Smooth the index fingertip position for a specific hand.

    Keep a small moving buffer for each detected hand so the cursor
    moves smoothly instead of jumping on every frame.
    """
    if hand_id not in finger_buffers:
        finger_buffers[hand_id] = deque(maxlen=SMOOTH_N)
    
    finger_buffers[hand_id].append((x_raw, y_raw))
    
    xs = [p[0] for p in finger_buffers[hand_id]]
    ys = [p[1] for p in finger_buffers[hand_id]]
    
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

# ============================================================================
# PLAY SOUND FUNCTION
# ============================================================================

def play_key(i: int, hand_id: int = 0):
    """Play a piano note if enough time has passed since the last press."""
    key_id = (hand_id, i)
    now = time.monotonic()
    last_time = last_played.get(key_id, 0)
    
    # Debounce so the same key does not re-trigger too quickly
    if now - last_time < DEBOUNCE_S:
        return
    
    if notes[i]:
        notes[i].stop()
        notes[i].play()
        print(f"🎵 Playing: {NOTE_NAMES[i]} (Hand {hand_id + 1})")
    
    last_played[key_id] = now
    key_lit[i] = now

# ============================================================================
# PINCH DETECTION
# ============================================================================

def is_pinching(hand_landmarks) -> bool:
    """Return True when the thumb and index finger tips are close together."""
    lm = hand_landmarks.landmark
    dx = lm[8].x - lm[4].x
    dy = lm[8].y - lm[4].y
    distance = (dx*dx + dy*dy) ** 0.5
    return distance < PINCH_THRESH

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def draw_piano_keys(frame):
    """Draw the piano keys with transparency and lit-key feedback."""
    now = time.monotonic()
    overlay = frame.copy()
    
    for i in range(NUM_KEYS):
        x1, y1, x2, y2 = key_rect(i)
        lit = (now - key_lit.get(i, -9)) < LIT_DURATION
        color = LIT_COLORS[i] if lit else KEY_COLORS[i]
        
        # Draw filled rectangle for the key background
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Draw border around each key for separation
        border_color = tuple(min(255, c + 60) for c in color)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, 2)
        
        # Draw note name centered on the key
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = NOTE_NAMES[i]
        text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
        tx = x1 + (KEY_W - text_size[0]) // 2
        ty = y1 + (y2 - y1) // 2 + text_size[1] // 2
        
        cv2.putText(overlay, text, (tx, ty), font, 0.6, (255, 255, 255), 2)
    
    # Blend the key overlay with the live camera frame
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw a horizontal divider line for the key zone
    cv2.line(frame, (0, KEY_ZONE_TOP), (CAM_W, KEY_ZONE_TOP), (255, 255, 255), 2)

def draw_pointer(frame, x, y, hand_id, is_pinching_val):
    """Draw the index-finger cursor and indicate pinch state."""
    if x is None or y is None:
        return
    
    # Choose pointer color by hand index
    if hand_id == 0:
        color = (0, 200, 255)  # Blue for first detected hand
    else:
        color = (255, 100, 100)  # Red for second detected hand
    
    if is_pinching_val:
        # Pinch mode shows rings around the cursor
        for r in range(20, 10, -2):
            cv2.circle(frame, (x, y), r, color, 2)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 6, color, -1)
    else:
        # Normal pointer style for hovering
        cv2.circle(frame, (x, y), 12, (0, 0, 0), 2)
        cv2.circle(frame, (x, y), 10, color, -1)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
    
    # Label the pointer with the hand number
    cv2.putText(frame, f"Hand {hand_id + 1}", (x - 20, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def draw_hud(frame, fps):
    """Draw HUD"""
    # Background
    cv2.rectangle(frame, (0, 0), (CAM_W, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (CAM_W, 50), (255, 255, 255), 1)
    
    # Title
    cv2.putText(frame, "AI VIRTUAL PIANO", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    
    # Mode
    mode_color = (0, 255, 0) if TRIGGER_MODE == "zone" else (255, 100, 0)
    cv2.putText(frame, f"Mode: {TRIGGER_MODE.upper()}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_color, 1)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps}", (CAM_W - 70, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Controls
    cv2.putText(frame, "Q:Quit M:Mode +/-:Speed", (CAM_W - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

# ============================================================================
# MAIN LOOP
# ============================================================================

print("=" * 60)
print("🎹 AI VIRTUAL PIANO - WORKING POINTER VERSION")
print("=" * 60)
print("Instructions:")
print("   • Show your hand to the camera")
print("   • Move your INDEX FINGER over the colored keys")
print("   • The blue/red circle will follow your finger")
print("   • Press 'M' to change modes")
print("   • Press 'Q' to quit")
print("=" * 60)

fps_start = time.time()
fps_counter = 0
fps_display = 0

while True:
    loop_start = time.time()
    
    # Read the next camera frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    
    # Mirror image so movements feel natural
    frame = cv2.flip(frame, 1)
    frame_count += 1
    fps_counter += 1
    
    # Update FPS display once per second
    if time.time() - fps_start >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        fps_start = time.time()
    
    # MediaPipe expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    
    # Run hand detection on the current frame
    results = hands_detector.process(rgb_frame)
    
    # Keep track of active hands this frame
    current_hands = {}
    
    # Draw the static piano keys before overlaying cursor graphics
    draw_piano_keys(frame)
    
    # Process each detected hand landmark set
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2:
                break
            
            # Draw hand skeleton
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(100, 100, 255), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1),
            )
            
            # GET INDEX FINGER TIP POSITION (Landmark 8)
            index_tip = hand_landmarks.landmark[8]
            
            # Convert normalized landmark coordinates to pixel coordinates
            fx_raw = int(index_tip.x * CAM_W)
            fy_raw = int(index_tip.y * CAM_H)
            
            # Debug circle at the exact index fingertip location
            cv2.circle(frame, (fx_raw, fy_raw), 5, (0, 255, 255), -1)
            
            # Smooth the raw finger position to reduce jitter
            fx_smooth, fy_smooth = smooth_point(hand_idx, fx_raw, fy_raw)
            
            # Detect whether the thumb and index finger are pinched together
            pinching = is_pinching(hand_landmarks)
            
            # Remember this hand for cleanup later
            current_hands[hand_idx] = (fx_smooth, fy_smooth, pinching)
            
            # Draw the on-screen cursor for this hand
            draw_pointer(frame, fx_smooth, fy_smooth, hand_idx, pinching)
            
            # Determine which key zone the finger is currently over
            current_in_zone = set()
            
            for key_i in range(NUM_KEYS):
                x1, y1, x2, y2 = key_rect(key_i)
                
                # True when the fingertip is inside the key's rectangle
                over = (x1 < fx_smooth < x2) and (y1 < fy_smooth < y2)
                
                if TRIGGER_MODE == "zone":
                    if over:
                        current_in_zone.add(key_i)
                        # Only trigger on the transition from outside to inside
                        if key_i not in prev_in_zone.get(hand_idx, set()):
                            play_key(key_i, hand_idx)
                else:  # pinch mode
                    if over and pinching:
                        play_key(key_i, hand_idx)
            
            # Save the current over-key state for the next frame
            prev_in_zone[hand_idx] = current_in_zone
            
            # Show the smoothed fingertip coordinates on screen
            cv2.putText(frame, f"Index Finger: ({fx_smooth}, {fy_smooth})", 
                       (10, CAM_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Clear inactive hands
    for hand_id in list(prev_in_zone.keys()):
        if hand_id not in current_hands:
            prev_in_zone[hand_id] = set()
            if hand_id in finger_buffers:
                finger_buffers[hand_id].clear()
    
    # Draw HUD
    draw_hud(frame, fps_display)
    
    # Show instruction
    if not results.multi_hand_landmarks:
        cv2.putText(frame, "✋ SHOW YOUR HAND TO THE CAMERA", 
                   (CAM_W//2 - 150, CAM_H//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Show the frame
    cv2.imshow("AI Virtual Piano - Move your Index Finger", frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        TRIGGER_MODE = "pinch" if TRIGGER_MODE == "zone" else "zone"
        print(f"🎵 Mode changed to: {TRIGGER_MODE.upper()}")
    elif key == ord('+') or key == ord('='):
        DEBOUNCE_S = min(DEBOUNCE_S + 0.02, 0.5)
        print(f"⚡ Speed: {int(DEBOUNCE_S*1000)}ms")
    elif key == ord('-'):
        DEBOUNCE_S = max(DEBOUNCE_S - 0.02, 0.05)
        print(f"⚡ Speed: {int(DEBOUNCE_S*1000)}ms")
    
    # Limit FPS
    elapsed = time.time() - loop_start
    if elapsed < 0.016:
        time.sleep(0.016 - elapsed)

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("\n👋 Thanks for playing! Goodbye!")