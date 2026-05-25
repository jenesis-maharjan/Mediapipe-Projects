import cv2
import mediapipe as mp
import pyautogui
import math
import time

# -------------------------
# CAMERA
# -------------------------
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

# -------------------------
# SCREEN SIZE
# -------------------------
screen_width, screen_height = pyautogui.size()

# Disable pyautogui's built-in pause
pyautogui.FAILSAFE = False
pyautogui.MINIMUM_DURATION = 0
pyautogui.PAUSE = 0

# -------------------------
# CALIBRATION - CHANGE THESE VALUES
# -------------------------
# Try different combinations until cursor moves correctly:
# Option 1: NORMAL - (1, 1) means no change
# Option 2: MIRROR HORIZONTAL - (-1, 1) 
# Option 3: MIRROR VERTICAL - (1, -1)
# Option 4: BOTH - (-1, -1)

X_DIRECTION = -1  # Change to 1 if cursor moves wrong direction
Y_DIRECTION = 1   # Change to -1 if up/down is reversed

# -------------------------
# MEDIAPIPE
# -------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# -------------------------
# SMOOTHING VARIABLES
# -------------------------
prev_x = 0
prev_y = 0
smoothening = 5

# Click debouncing
last_click_time = 0
click_cooldown = 0.15

print("🖱️ Virtual Mouse Started")
print("Press Q to quit")
print("\n--- CALIBRATION ---")
print("If cursor moves opposite direction:")
print("- Change X_DIRECTION to 1 or -1")
print("- Change Y_DIRECTION to 1 or -1")
print(f"\nCurrent settings:")
print(f"X_DIRECTION = {X_DIRECTION} (1=normal, -1=reversed)")
print(f"Y_DIRECTION = {Y_DIRECTION} (1=normal, -1=reversed)")

while True:
    success, frame = cap.read()

    if not success:
        break

    # Optional: Uncomment to flip the DISPLAY only (not cursor)
    # frame = cv2.flip(frame, 1)
    
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark

        # -------------------------
        # INDEX FINGER TIP
        # -------------------------
        index_tip = landmarks[8]

        ix = int(index_tip.x * w)
        iy = int(index_tip.y * h)

        # -------------------------
        # MIDDLE FINGER TIP
        # -------------------------
        middle_tip = landmarks[12]

        mx = int(middle_tip.x * w)
        my = int(middle_tip.y * h)

        # Draw circles
        cv2.circle(frame, (ix, iy), 10, (0,255,0), -1)
        cv2.circle(frame, (mx, my), 10, (255,0,0), -1)

        # -------------------------
        # MOVE MOUSE - WITH DIRECTION CONTROL
        # -------------------------
        # Apply direction multipliers
        if X_DIRECTION == 1:
            screen_x = screen_width * index_tip.x
        else:
            screen_x = screen_width * (1 - index_tip.x)
            
        if Y_DIRECTION == 1:
            screen_y = screen_height * index_tip.y
        else:
            screen_y = screen_height * (1 - index_tip.y)

        # Smoothing
        curr_x = prev_x + (screen_x - prev_x) / smoothening
        curr_y = prev_y + (screen_y - prev_y) / smoothening

        # Constrain to screen boundaries
        curr_x = max(0, min(screen_width, curr_x))
        curr_y = max(0, min(screen_height, curr_y))

        pyautogui.moveTo(curr_x, curr_y, duration=0)

        prev_x = curr_x
        prev_y = curr_y

        # -------------------------
        # CLICK DETECTION
        # -------------------------
        distance = math.hypot(mx - ix, my - iy)

        cv2.line(frame, (ix, iy), (mx, my), (255,255,255), 2)

        current_time = time.time()
        
        if distance < 35 and (current_time - last_click_time) > click_cooldown:
            cv2.putText(frame, "CLICK!", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

            pyautogui.click()
            last_click_time = current_time

        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )
        
        # Show current direction settings on screen
        cv2.putText(frame, f"X_DIR: {X_DIRECTION}  Y_DIR: {Y_DIRECTION}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("Virtual Mouse", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Hotkeys to change direction while running
    elif key == ord('x'):
        X_DIRECTION *= -1
        print(f"X_DIRECTION changed to {X_DIRECTION}")
    elif key == ord('y'):
        Y_DIRECTION *= -1
        print(f"Y_DIRECTION changed to {Y_DIRECTION}")

cap.release()
cv2.destroyAllWindows()