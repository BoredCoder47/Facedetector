import os
import face_recognition
import numpy as np
import threading
import cv2  # for reliable BGR → RGB conversion


class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces", tolerance=0.60):
        """
        Fast face recognition with caching and thread safety.

        known_faces_dir:
            Folder containing reference face images (jpg/png/jpeg)
        tolerance:
            Lower = stricter matching (default 0.45)
        """
        self.known_encodings = []
        self.known_names = []
        self.tolerance = tolerance
        self.lock = threading.Lock()

        self._load_known_faces(known_faces_dir)

    # -----------------------------
    # Load known faces from directory
    # -----------------------------
    def _load_known_faces(self, known_faces_dir):
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"[FaceRecognizer] Created empty directory: {known_faces_dir}")
            return

        print(f"[FaceRecognizer] Loading faces from '{known_faces_dir}'...")
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(known_faces_dir, filename)
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    self.known_encodings.append(encodings[0])
                    self.known_names.append(os.path.splitext(filename)[0])
                    print(f"[FaceRecognizer] Loaded: {filename}")
                else:
                    print(f"[FaceRecognizer] WARNING: No face found in {filename}")

        print(f"[FaceRecognizer] Loaded {len(self.known_names)} known faces.")

    # -----------------------------
    # Normalize and convert face boxes
    # -----------------------------
    def _normalize_boxes(self, boxes, frame_shape):
        """
        Ensures face_boxes are in (top, right, bottom, left) format.
        Accepts both absolute and relative (0–1) coords.
        """
        h, w = frame_shape[:2]
        normalized = []

        for box in boxes:
            # Handle dicts from MediaPipe or other detectors
            if isinstance(box, dict) and "x" in box:
                x, y, width, height = box["x"], box["y"], box["width"], box["height"]
                top = int(y * h) if y <= 1 else int(y)
                left = int(x * w) if x <= 1 else int(x)
                bottom = int((y + height) * h) if height <= 1 else int(y + height)
                right = int((x + width) * w) if width <= 1 else int(x + width)

            # Handle tuple/list (common: (top, right, bottom, left) or (ymin, xmin, ymax, xmax))
            elif isinstance(box, (list, tuple)) and len(box) == 4:
                # Detect if order is likely (ymin, xmax, ymax, xmin)
                ymin, xmax, ymax, xmin = box
                top = int(ymin)
                right = int(xmax)
                bottom = int(ymax)
                left = int(xmin)

            else:
                continue  # Skip invalid formats

            # Clamp to image size
            top = max(0, min(top, h))
            right = max(0, min(right, w))
            bottom = max(0, min(bottom, h))
            left = max(0, min(left, w))

            normalized.append((top, right, bottom, left))

        return normalized

    # -----------------------------
    # Recognize faces
    # -----------------------------
    def recognize(self, frame, face_boxes, expected_user=None):
        """
        Recognize faces using given bounding boxes.
        Returns:
            [{"name": <name>, "location": <bbox>, "expected_user": <expected_user>}]
        """
        results = []

        if frame is None or not face_boxes:
            return results

        # Convert BGR → RGB reliably
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ensure correct format for Dlib
        face_boxes = self._normalize_boxes(face_boxes, frame.shape)

        if not face_boxes:
            return results

        # Compute encodings
        try:
            encodings = face_recognition.face_encodings(frame, known_face_locations=face_boxes)
        except Exception as e:
            print(f"[FaceRecognizer] Encoding error: {e}")
            return results

        for encoding, (top, right, bottom, left) in zip(encodings, face_boxes):
            name = "Unknown"
            if self.known_encodings:
                with self.lock:
                    distances = face_recognition.face_distance(self.known_encodings, encoding)
                    best_idx = np.argmin(distances)
                    if distances[best_idx] <= self.tolerance:
                        name = self.known_names[best_idx]

            if expected_user:
                print(f"[FaceRecognizer] Detected: {name} | Expected: {expected_user}")

            results.append({
                "name": name,
                "location": (top, right, bottom, left),
                "expected_user": expected_user
            })

        return results
