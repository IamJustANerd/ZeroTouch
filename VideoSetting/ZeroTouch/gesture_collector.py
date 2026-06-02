import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import csv
import os

class GestureCollector:
    """
    Extends hand tracking to save normalized landmark data for training.
    """
    def __init__(self, model_path='hand_landmarker.task', csv_file='gesture_data.csv'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self.csv_file = csv_file

    def process_frame(self, img, timestamp_ms):
        """
        Detects hands and provides visual feedback.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if self.results.hand_landmarks:
            for landmarks in self.results.hand_landmarks:
                h, w, _ = img.shape
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return img

    def save_data(self, label):
        """
        Saves relative landmark coordinates to CSV.
        Format: label, dx0, dy0, dz0... dx20, dy20, dz20 (Wrist is the anchor)
        """
        if not self.results or not self.results.hand_landmarks:
            print("No hands detected. Data not saved.")
            return False

        # Use the first hand detected
        landmarks = self.results.hand_landmarks[0]
        
        # Normalize: Subtract wrist (id 0) from all landmarks
        wrist = landmarks[0]
        data = [label] 
        for lm in landmarks:
            # We store x, y, and z relative to the wrist
            data.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        # Append to CSV file
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
        
        print(f"Saved gesture '{label}' to {self.csv_file}")
        return True

def main():
    cap = cv2.VideoCapture(0)
    collector = GestureCollector()
    start_time = time.time()

    print("--- Gesture Data Collector ---")
    print("1. Show your hand pose.")
    print("2. Press 0-9 to save with a label ID.")
    print("   0: Open Palm, 1: Fist, 2: Pinch, 3: Pointing")
    print("3. Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success: break

        timestamp_ms = int((time.time() - start_time) * 1000)
        img = collector.process_frame(img, timestamp_ms)

        cv2.imshow("ZeroTouch: Data Collector", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif ord('0') <= key <= ord('9'):
            label = chr(key)
            collector.save_data(label)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
