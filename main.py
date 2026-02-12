import os
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import cv2
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt


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

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


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

            if detection_result.face_landmarks:
                # Face landmarks is a list of lists; we take the first face
                landmarks = detection_result.face_landmarks[0]
                frame = draw_landmarks_on_image(frame, detection_result)
                
            if detection_result.face_blendshapes:
                plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
                # for category in detection_result.face_blendshapes[0]:
                #     if category.category_name in ['mouthSmileLeft', 'mouthSmileRight']:
                #         # This value ranges from 0.0 to 1.0
                #         if category.score > 0.5: 
                #              cv2.putText(frame, f"{category.category_name}: {category.score:.2f}", 
                #                         (50, 80 if 'Left' in category.category_name else 110), 
                #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

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