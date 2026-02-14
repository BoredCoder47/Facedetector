import cv2
import time
import os
import threading

print("CWD:", os.getcwd())

from dotenv import load_dotenv
from urllib.parse import urlparse
from workers.frame_processor import FrameProcessor
# from workers.audio_processor import AudioProcessor  # 🔇 AUDIO DISABLED


# -----------------------------
# Database Configuration
# -----------------------------
env_path = os.path.join(os.getcwd(), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
else:
    print(f".env not found at: {env_path}")

DATABASE_URL = os.getenv("DATABASE_URL")
print("Loaded DATABASE_URL:", DATABASE_URL)

if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL not found in .env file.")

url = urlparse(DATABASE_URL)

db_config = {
    "host": url.hostname,
    "dbname": url.path[1:],
    "user": url.username,
    "password": url.password,
    "port": url.port or 5432,
    "sslmode": "require",
}


# -----------------------------
# Session ID
# -----------------------------
shared_session_id = f"session_{int(time.time())}"

video_processor = FrameProcessor(
    db_config=db_config,
    expected_user="photo1",
    session_id=shared_session_id
)

print("Session ID:", shared_session_id)


# -----------------------------
# Live Camera + Background Detection
# -----------------------------

# Force faster Windows backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Failed to open webcam.")

# Reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Optional FPS cap
cap.set(cv2.CAP_PROP_FPS, 20)

print("Press ESC to exit")


latest_alerts = []
shared_frame = None
lock = threading.Lock()
running = True


def detection_worker():
    global latest_alerts
    while running:
        if shared_frame is None:
            continue

        # Run detection on a copy (prevents race issues)
        anomalies = video_processor.process_frame(shared_frame.copy())

        alerts = []
        for anomaly in anomalies:
            if anomaly.get("imposter_detected"):
                alerts.append("IMPOSTER DETECTED")
            if anomaly.get("looking_away"):
                alerts.append("LOOKING AWAY")
            if anomaly.get("multiple_faces"):
                alerts.append("MULTIPLE FACES")

        with lock:
            latest_alerts = alerts


# Start background detection
thread = threading.Thread(target=detection_worker, daemon=True)
thread.start()


try:
    while True:
        # Grab newest frame (low latency)
        if not cap.grab():
            continue

        ret, frame = cap.retrieve()
        if not ret:
            continue

        # Update shared frame for background thread
        shared_frame = frame

        # Draw latest alerts without blocking
        with lock:
            alerts_to_draw = latest_alerts.copy()

        if alerts_to_draw:
            cv2.putText(
                frame,
                " | ".join(set(alerts_to_draw)),
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # Show immediately (no waiting for detection)
        cv2.imshow("Exam Proctoring Prototype", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Keyboard interrupt received, stopping...")

finally:
    running = False
    thread.join(timeout=2)

    cap.release()
    cv2.destroyAllWindows()
    video_processor.close()

    print("✅ Clean exit.")
