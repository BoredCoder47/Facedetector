import cv2

class GazeDetector:
    def __init__(self, threshold=0.35, debug=False):
        """
        threshold: fraction of FRAME width/height considered as central focus zone.
        (0.35 → roughly central 30% area is valid)
        debug: if True, draws the central focus zone for visualization.
        """
        self.threshold = threshold
        self.debug = debug

    def is_looking_away(self, left_center, right_center, frame_width, frame_height, frame=None):
        """
        Returns True if both eyes are outside the central viewing zone.
        Optionally draws the zone if `debug=True` and frame is provided.
        """
        # Handle invalid or missing eye landmarks
        if not left_center or not right_center or frame_width == 0 or frame_height == 0:
            return True  # Consider as looking away

        # Normalize positions to [0, 1]
        left_x, left_y = left_center[0] / frame_width, left_center[1] / frame_height
        right_x, right_y = right_center[0] / frame_width, right_center[1] / frame_height

        # Define central focus zone
        x_min, x_max = self.threshold, 1 - self.threshold
        y_min, y_max = self.threshold, 1 - self.threshold

        # Check if both eyes are outside the zone
        left_out = not (x_min <= left_x <= x_max and y_min <= left_y <= y_max)
        right_out = not (x_min <= right_x <= x_max and y_min <= right_y <= y_max)

        # Optional visualization for debugging
        if self.debug and frame is not None:
            self._draw_focus_zone(frame, x_min, x_max, y_min, y_max, frame_width, frame_height)

        return left_out and right_out

    def _draw_focus_zone(self, frame, x_min, x_max, y_min, y_max, w, h):
        """Draw the central focus zone as a rectangle overlay (for debugging)."""
        start_point = (int(x_min * w), int(y_min * h))
        end_point = (int(x_max * w), int(y_max * h))
        cv2.rectangle(frame, start_point, end_point, (0, 255, 255), 2)
