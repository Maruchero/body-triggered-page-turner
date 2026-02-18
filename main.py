import cv2
import sys
import os
import time # Added for keypress cooldown

# Set environment variables for Qt if needed
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np

from PySide6 import QtWidgets
import pyqtgraph as pg
import pyautogui # Added for simulating key presses


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


class BlendshapeVisualizer:
    def __init__(self):
        self.win = pg.GraphicsLayoutWidget(show=True, title="Face Blendshapes")
        self.win.resize(800, 400)
        self.plot = self.win.ci.addPlot(title="Mouth Blendshapes Time Series")
        self.plot.addLegend()
        self.plot.getViewBox().setYRange(0, 1)
        self.plot.setLabel("left", "Score")
        self.plot.setLabel("bottom", "Frames (Last 200)")

        self.whitelist = [
            "mouthLowerDownLeft",
            "mouthLowerDownRight",
            "mouthSmileLeft",
            "mouthSmileRight",
        ]
        self.history_size = 200
        self.data = {name: np.zeros(self.history_size) for name in self.whitelist}
        self.curves = {}

        # Define some distinct colors for the lines
        colors = ["r", "g", "b", "y"]
        for i, name in enumerate(self.whitelist):
            self.curves[name] = self.plot.plot(
                pen=pg.mkPen(colors[i % len(colors)], width=2), name=name
            )
        
        # Initialize keypress cooldown
        self.last_keypress_time = 0
        self.keypress_cooldown = 1 # seconds


    def update(self, face_blendshapes):
        for b in face_blendshapes:
            if b.category_name in self.whitelist:
                # Shift data to the left and add the new score at the end
                self.data[b.category_name][:-1] = self.data[b.category_name][1:]
                self.data[b.category_name][-1] = b.score
                # Update the corresponding curve
                self.curves[b.category_name].setData(self.data[b.category_name])
            
        # Calculate overall smile score as average of left and right
        smile_score = (self.data["mouthSmileLeft"][-1] + self.data["mouthSmileRight"][-1]) / 2.0

        # Check for keypress condition with cooldown
        if smile_score > 0.85 and (time.time() - self.last_keypress_time > self.keypress_cooldown):
            print("[DEBUG] Smile score exceeded threshold: {:.2f}".format(smile_score))
            pyautogui.press('right')
            self.last_keypress_time = time.time()


def initialize_smile_turner(index=0):
    # Initialize Qt Application safely
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    visualizer = BlendshapeVisualizer()

    window_name = "Smile Turner - Task API Check"
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print(f"Error: Could not open video device at index {index}.")
        sys.exit(1)

    # STEP 1: Create the FaceLandmarker detector
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found in the directory.")
        sys.exit(1)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, output_face_blendshapes=True, num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    print("MediaPipe Tasks loaded. Press 'q' or [X] to exit.")

    try:
        while True:
            # Process Qt events to keep the graph responsive
            app.processEvents()

            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = detector.detect(mp_image)

            if detection_result.face_landmarks:
                frame = draw_landmarks_on_image(frame, detection_result)

            if detection_result.face_blendshapes and visualizer.win.isVisible():
                visualizer.update(detection_result.face_blendshapes[0])

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                visualizer.win.close()
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    visualizer.win.close()
                    break
            except cv2.error:
                visualizer.win.close()
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    initialize_smile_turner()

