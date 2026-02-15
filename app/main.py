import os
import time
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from urllib.parse import urlparse
from workers.frame_processor import FrameProcessor
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL not found in environment.")

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
# FastAPI App
# -----------------------------
app = FastAPI(title="Exam Proctoring Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # demo only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Initialize FrameProcessor ONCE
# -----------------------------
shared_session_id = f"session_{int(time.time())}"

video_processor = FrameProcessor(

    expected_user="photo1",
    session_id=shared_session_id
)

print("Session ID:", shared_session_id)


# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.get("/")
def health():
    return {"status": "running", "session_id": shared_session_id}


# -----------------------------
# Frame Processing Endpoint
# -----------------------------
@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Invalid image file."}

        anomalies = video_processor.process_frame(frame)

        return {
            "session_id": shared_session_id,
            "anomalies": anomalies
        }

    except Exception as e:
        return {"error": str(e)}
