import os
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import cv2
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def initialize_smile_turner(index=0):
    window_name = 'Smile Turner - Task API Check'
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print(f"Error: Could not open video device at index {index}.")
        sys.exit(1)

    # STEP 1: Create the FaceLandmarker detector
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found in the directory.")
        sys.exit(1)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True, # We will use this for the smile later!
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    print("MediaPipe Tasks loaded. Press 'q' or [X] to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # STEP 2: Convert the frame to MediaPipe's Image format
            # Convert BGR to RGB first
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # STEP 3: Detect landmarks
            detection_result = detector.detect(mp_image)

            # For now, let's just print if a face is detected
            if detection_result.face_landmarks:
                # We will add drawing logic in the next iteration 
                # to keep this specific step clean.
                cv2.putText(frame, "Face Detected", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    initialize_smile_turner()