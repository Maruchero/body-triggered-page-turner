import cv2
import sys
import os

# Set environment variables for Qt if needed
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np

# Replace matplotlib with pyqtgraph
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg

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
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

  return annotated_image

class BlendshapeVisualizer:
    def __init__(self):
        self.win = pg.GraphicsLayoutWidget(show=True, title="Face Blendshapes")
        self.win.resize(800, 600)
        self.plot = self.win.addPlot(title="Blendshape Scores")
        
        self.bar_graph = None
        self.category_names = []
        
    def update(self, face_blendshapes):
        scores = [b.score for b in face_blendshapes]
        names = [b.category_name for b in face_blendshapes]
        
        if self.bar_graph is None:
            self.category_names = names
            self.bar_graph = pg.BarGraphItem(x0=0, y=np.arange(len(scores)), height=0.6, width=scores, brush='b')
            self.plot.addItem(self.bar_graph)
            
            # Setup Y-axis labels
            ay = self.plot.getAxis('left')
            ticks = [(i, name) for i, name in enumerate(names)]
            ay.setTicks([ticks])
            self.plot.setYRange(-1, len(names))
            self.plot.setXRange(0, 1)
            self.plot.invertY(True)
        else:
            self.bar_graph.setOpts(width=scores)

def initialize_smile_turner(index=0):
    # Initialize Qt Application
    app = QtWidgets.QApplication(sys.argv)
    visualizer = BlendshapeVisualizer()

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
        output_face_blendshapes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    print("MediaPipe Tasks loaded. Press 'q' or [X] to exit.")

    try:
        while True:
            # Process Qt events to keep the graph responsive
            app.processEvents()

            ret, frame = cap.read()
            if not ret: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = detector.detect(mp_image)

            if detection_result.face_landmarks:
                frame = draw_landmarks_on_image(frame, detection_result)
                
            if detection_result.face_blendshapes:
                visualizer.update(detection_result.face_blendshapes[0])

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

if __name__ == "__main__":
    initialize_smile_turner()