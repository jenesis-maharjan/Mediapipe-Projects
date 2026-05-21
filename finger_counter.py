import cv2  # OpenCV for webcam and image processing
import mediapipe as mp  # MediaPipe for hand tracking

class FingerCounter:
    def __init__(self):
        # Initialize MediaPipe Hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,      # For real-time video (not static images)
            max_num_hands=1,              # Track only one hand for simplicity
            min_detection_confidence=0.7, # Minimum confidence for detection
            min_tracking_confidence=0.5   # Minimum confidence for tracking
        )
        self.mp_draw = mp.solutions.drawing_utils  # Utility to draw landmarks on hand

        # MediaPipe hand landmark indices (0-20)
        self.finger_tips = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky tips
        # FIX: Use PIP joints (middle knuckle) instead of MCP (base knuckle)
        # for more accurate "finger is up" detection
        self.finger_pip = [3, 6, 10, 14, 18]    # PIP joints (one below the tip)
        self.finger_mcp = [2, 5, 9, 13, 17]     # MCP joints (kept for thumb reference)

    def count_fingers(self, hand_landmarks, handedness):
        """
        Count how many fingers are extended/raised.

        Parameters:
            hand_landmarks: Detected hand landmarks from MediaPipe
            handedness: "Right" or "Left" label from MediaPipe (in mirrored frame)

        Returns:
            int: Number of extended fingers (0-5)
        """
        fingers = []
        landmarks = hand_landmarks.landmark

        # THUMB detection
        # After horizontal flip, MediaPipe still labels hands as "Right"/"Left"
        # based on the original (unflipped) view, so we must invert the logic:
        #   - "Right" hand in mirrored frame → thumb tip is to the LEFT of MCP when open
        #   - "Left" hand in mirrored frame  → thumb tip is to the RIGHT of MCP when open
        # FIX: account for handedness so thumb is detected correctly after flip
        if handedness == "Right":
            # Mirrored right hand: tip.x < mcp.x means thumb is extended
            fingers.append(1 if landmarks[self.finger_tips[0]].x < landmarks[self.finger_mcp[0]].x else 0)
        else:
            # Mirrored left hand: tip.x > mcp.x means thumb is extended
            fingers.append(1 if landmarks[self.finger_tips[0]].x > landmarks[self.finger_mcp[0]].x else 0)

        # Check other 4 fingers (Index, Middle, Ring, Pinky)
        # FIX: Compare tip Y to PIP Y (not MCP) for more reliable detection
        # Tip must be higher (smaller Y) than the PIP joint to count as extended
        for i in range(1, 5):
            if landmarks[self.finger_tips[i]].y < landmarks[self.finger_pip[i]].y:
                fingers.append(1)  # Finger is extended
            else:
                fingers.append(0)  # Finger is folded

        return sum(fingers)

    def draw_finger_status(self, frame, hand_landmarks, finger_count, handedness):
        """
        Draw finger count, gesture label, and highlight fingertips on the frame.

        Parameters:
            frame: The image/frame to draw on
            hand_landmarks: Detected hand landmarks
            finger_count: Number of extended fingers
            handedness: "Right" or "Left"
        """
        h, w = frame.shape[:2]
        landmarks = hand_landmarks.landmark

        # Draw a semi-transparent background panel at top-left
        cv2.rectangle(frame, (10, 10), (280, 110), (0, 0, 0), -1)   # Filled black box
        cv2.rectangle(frame, (10, 10), (280, 110), (255, 255, 255), 2)  # White border

        # Display finger count and hand label
        cv2.putText(frame, f"Fingers: {finger_count}  ({handedness} hand)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Gesture label based on finger count
        gesture_map = {
            0: "✊ Fist",
            1: "☝️ Pointing",
            2: "✌️ Peace Sign",
            3: "🤟 Three",
            4: "🖖 Four",
            5: "🖐️ Open Hand",
        }
        gesture = gesture_map.get(finger_count, f"{finger_count} Fingers")
        cv2.putText(frame, gesture, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Color-code each fingertip: green = extended, red = folded
        finger_names = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]

        for i, tip_id in enumerate(self.finger_tips):
            tip = landmarks[tip_id]
            x, y = int(tip.x * w), int(tip.y * h)

            # FIX: mirror-aware thumb check, consistent with count_fingers()
            if i == 0:
                if handedness == "Right":
                    is_extended = tip.x < landmarks[self.finger_mcp[i]].x
                else:
                    is_extended = tip.x > landmarks[self.finger_mcp[i]].x
            else:
                is_extended = tip.y < landmarks[self.finger_pip[i]].y

            color = (0, 255, 0) if is_extended else (0, 0, 255)

            cv2.circle(frame, (x, y), 8, color, -1)           # Filled circle
            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)  # White border
            cv2.putText(frame, finger_names[i], (x - 20, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def run(self):
        """Main function to run the finger counter with webcam feed."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("=" * 50)
        print("FINGER COUNTER USING MEDIAPIPE")
        print("=" * 50)
        print("Instructions:")
        print("- Show your hand to the camera")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset screenshot counter")
        print("- Press 's' to save screenshot")
        print("=" * 50)

        frame_count = 0
        screenshot_count = 1

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert BGR → RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True  # (not used further; BGR frame is drawn on)

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]

                # FIX: read handedness label so thumb logic can use it
                handedness = results.multi_handedness[0].classification[0].label

                # Draw skeleton
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                finger_count = self.count_fingers(hand_landmarks, handedness)
                self.draw_finger_status(frame, hand_landmarks, finger_count, handedness)

                # Gesture-specific banner
                if finger_count == 2:
                    cv2.putText(frame, "PEACE SIGN DETECTED!", (320, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                elif finger_count == 5:
                    cv2.putText(frame, "HIGH FIVE!", (320, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            else:
                cv2.putText(frame, "No hand detected", (320, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Show your hand to the camera", (320, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Bottom status bar
            cv2.putText(frame, f"Frame: {frame_count}", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, "Press 'q' to quit | 's' for screenshot", (10, 475),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow('Finger Counter - Show Your Hand', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting program...")
                break
            elif key == ord('s'):
                name = f"finger_counter_{screenshot_count}.png"
                cv2.imwrite(name, frame)
                print(f"Screenshot saved as '{name}'")
                screenshot_count += 1
            elif key == ord('r'):
                print("\nResetting screenshot counter...")
                screenshot_count = 1

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print("Program terminated successfully!")


if __name__ == "__main__":
    counter = FingerCounter()
    counter.run()