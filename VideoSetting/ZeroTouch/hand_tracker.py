import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import tensorflow as tf
import pyautogui
from collections import deque
import sys
import os
import ctypes

def get_asset_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# PERFORMANCE TWEAKS:
pyautogui.PAUSE = 0 
pyautogui.FAILSAFE = True 

class ClutchManager:
    """
    Manages the 3-state safety system: IDLE -> LISTENING -> ACTIVE.
    """
    def __init__(self, wake_threshold=2.0, grace_period=1.0):
        self.state = 'IDLE' 
        self.wake_threshold = wake_threshold
        self.grace_period = grace_period 
        self.start_hold_time = None
        self.lost_hand_time = None
        self.progress = 0 

    def update(self, current_gesture):
        if current_gesture == "Open Palm":
            self.lost_hand_time = None 
            if self.state == 'IDLE':
                self.state = 'LISTENING'
                self.start_hold_time = time.time()
            elif self.state == 'LISTENING':
                elapsed = time.time() - self.start_hold_time
                self.progress = min(elapsed / self.wake_threshold, 1.0)
                if elapsed >= self.wake_threshold:
                    self.state = 'ACTIVE'
                    self.progress = 0
        
        elif current_gesture in ["No Hand", "Unknown"]:
            if self.state != 'IDLE':
                if self.lost_hand_time is None:
                    self.lost_hand_time = time.time()
                if time.time() - self.lost_hand_time > self.grace_period:
                    self.state = 'IDLE'
                    self.progress = 0
                    self.lost_hand_time = None
        
        else:
            self.lost_hand_time = None 
            if self.state == 'LISTENING':
                self.state = 'IDLE'
                self.progress = 0

        return self.state, self.progress

    def reset(self):
        self.state = 'LISTENING'
        self.start_hold_time = time.time()
        self.progress = 0

class HandTracker:
    def __init__(self, landmarker_path='hand_landmarker.task', 
                 mlp_path='gesture_classifier.tflite'):
        
        landmarker_path = get_asset_path(landmarker_path)
        mlp_path = get_asset_path(mlp_path)
        
        base_options = python.BaseOptions(model_asset_path=landmarker_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.mlp_interpreter = tf.lite.Interpreter(model_path=mlp_path)
        self.mlp_interpreter.allocate_tensors()
        self.mlp_input = self.mlp_interpreter.get_input_details()
        self.mlp_output = self.mlp_interpreter.get_output_details()
        self.mlp_labels = ['Open Palm', 'Fist', 'Pinch', 'Pointing']
        
        self.results = None
        self.screen_width, self.screen_height = pyautogui.size()
        self.click_active = False 
        self.last_pinch_dist = None 
        
        # EMA Smoothing Filter
        self.smooth_x = self.screen_width // 2
        self.smooth_y = self.screen_height // 2
        self.alpha = 0.4 
        self.max_pinch_delta = 0.05 # Noise filter: ignore jumps > 5% screen

    def find_hands(self, img, timestamp_ms, draw=True):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
        if draw and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                self._draw_landmarks(img, hand_landmarks)
        return img

    def _draw_landmarks(self, img, landmarks):
        h, w, _ = img.shape
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

    def classify_gestures(self):
        if not self.results or not self.results.hand_landmarks:
            return "No Hand"

        landmarks = self.results.hand_landmarks[0]
        wrist = landmarks[0]
        
        mlp_data = []
        for lm in landmarks:
            mlp_data.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        
        mlp_input_data = np.array([mlp_data], dtype=np.float32)
        self.mlp_interpreter.set_tensor(self.mlp_input[0]['index'], mlp_input_data)
        self.mlp_interpreter.invoke()
        mlp_out = self.mlp_interpreter.get_tensor(self.mlp_output[0]['index'])
        
        prediction = self.mlp_labels[np.argmax(mlp_out)] if mlp_out[0][np.argmax(mlp_out)] > 0.8 else "Unknown"
        return prediction

    def interact_os(self, mlp_gesture):
        if not self.results or not self.results.hand_landmarks:
            self.last_pinch_dist = None
            return

        landmarks = self.results.hand_landmarks[0]
        h, w = landmarks[9].y, landmarks[9].x 
        
        raw_x = np.interp(w, [0.2, 0.8], [10, self.screen_width - 10])
        raw_y = np.interp(h, [0.2, 0.8], [10, self.screen_height - 10])
        
        prev_x = self.smooth_x
        prev_y = self.smooth_y
        
        self.smooth_x = self.alpha * raw_x + (1 - self.alpha) * self.smooth_x
        self.smooth_y = self.alpha * raw_y + (1 - self.alpha) * self.smooth_y

        if mlp_gesture == "Open Palm":
            pyautogui.moveTo(self.smooth_x, self.smooth_y)
            pyautogui.mouseUp()
            self.click_active = False
            self.last_pinch_dist = None

        elif mlp_gesture == "Pointing":
            if not self.click_active:
                pyautogui.click()
                self.click_active = True
            self.last_pinch_dist = None

        elif mlp_gesture == "Fist":
            pyautogui.mouseDown()
            dx = int(self.smooth_x - prev_x)
            dy = int(self.smooth_y - prev_y)
            ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0) # MOUSEEVENTF_MOVE
            self.click_active = False
            self.last_pinch_dist = None

        elif mlp_gesture == "Pinch":
            # Universal Proportional Zoom
            t_tip = landmarks[4]
            i_tip = landmarks[8]
            curr_dist = np.sqrt((t_tip.x - i_tip.x)**2 + (t_tip.y - i_tip.y)**2)

            if self.last_pinch_dist is not None:
                diff = curr_dist - self.last_pinch_dist
                # Sensitivity 8000. Big finger move -> big scroll.
                if abs(diff) < self.max_pinch_delta:
                    scroll_amount = int(diff * 8000) 
                    if abs(scroll_amount) > 10: 
                        pyautogui.scroll(scroll_amount)
            
            self.last_pinch_dist = curr_dist
            self.click_active = False

        else:
            self.click_active = False
            self.last_pinch_dist = None

class HandTrackerEngine:
    def __init__(self, debug=True, callback=None):
        self.tracker = HandTracker()
        self.clutch = ClutchManager()
        self.debug = debug
        self.callback = callback
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(0)
        p_time = 0
        start_time = time.time()
        self.running = True

        window_name = "ZeroTouch Engine"
        win_w, win_h = 320, 240
        
        if not self.callback:
            cv2.namedWindow(window_name)
            screen_w, screen_h = pyautogui.size()
            cv2.moveWindow(window_name, screen_w - win_w - 50, screen_h - win_h - 100)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        while self.running:
            success, img = cap.read()
            if not success: break
            img = cv2.flip(img, 1)

            timestamp_ms = int((time.time() - start_time) * 1000)
            img = self.tracker.find_hands(img, timestamp_ms)
            mlp_gest = self.tracker.classify_gestures()
            state, progress = self.clutch.update(mlp_gest)

            if state == 'ACTIVE':
                self.tracker.interact_os(mlp_gest)

            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time

            if self.callback:
                self.callback(img, state, progress, mlp_gest, fps)
            else:
                display_img = cv2.resize(img, (win_w, win_h))
                if not self.debug:
                    display_img = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    cv2.putText(display_img, f"STATUS: {state}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.imshow(window_name, display_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

def main(debug=True):
    engine = HandTrackerEngine(debug=debug)
    engine.run()

if __name__ == "__main__":
    main()
