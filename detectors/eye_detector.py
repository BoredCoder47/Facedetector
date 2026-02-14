import mediapipe as mp
import cv2
import numpy as np

class EyeDetector:
    def __init__(self):
        # Initialize FaceMesh with lighter configuration for real-time tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            static_image_mode=False,       # <-- Enables temporal tracking
            min_detection_confidence=0.5,  # slightly lower to reduce missed detections
            min_tracking_confidence=0.6
        )

        # Eye landmark indices
        self.LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
        self.RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

    def get_eye_landmarks(self, frame):
        """
        Fast eye landmark detection with optional downscaling for performance.
        Returns (left_eye, right_eye, face_landmarks)
        """
        if frame is None or frame.size == 0:
            return None, None, None

        # --- Downscale frame to speed up detection ---
        h, w, _ = frame.shape
        scale_factor = 0.5  # tweak: 0.4–0.6 for best balance
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_small)

        if not results.multi_face_landmarks:
            return None, None, None

        # Convert coordinates back to original scale
        face_landmarks = results.multi_face_landmarks[0]
        left_eye = [
            (
                int(face_landmarks.landmark[i].x * w),
                int(face_landmarks.landmark[i].y * h)
            )
            for i in self.LEFT_EYE_IDX
        ]
        right_eye = [
            (
                int(face_landmarks.landmark[i].x * w),
                int(face_landmarks.landmark[i].y * h)
            )
            for i in self.RIGHT_EYE_IDX
        ]

        return left_eye, right_eye, face_landmarks

    @staticmethod
    def compute_eye_center(eye_landmarks):
        """Compute mean (x, y) of landmarks efficiently using NumPy."""
        if not eye_landmarks:
            return None
        pts = np.array(eye_landmarks, dtype=np.float32)
        return tuple(np.mean(pts, axis=0))
