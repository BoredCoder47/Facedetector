import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv
import cv2
import tempfile
import traceback

# Load Cloudinary credentials
load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,  # ✅ ensures https URLs
    
)


class CloudinaryUploader:
    @staticmethod
    def upload_frame(frame, session_id, frame_id, anomaly_reason):
        """
        Uploads a frame (numpy array) to Cloudinary.
        Returns the secure URL of the uploaded image.
        """
        if frame is None or frame.size == 0:
            print("[CloudinaryUploader] Skipped: empty or invalid frame.")
            return None

        tmp_path = None
        try:
            # ✅ Save frame temporarily as an image
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                success = cv2.imwrite(tmp_path, frame)
                if not success:
                    raise ValueError("cv2.imwrite() failed to save frame.")
            print("trying to upload")   
            # ✅ Upload image to Cloudinary
            upload_result = cloudinary.uploader.upload(
                tmp_path,
                folder=f"anomaly_logs/{session_id}",
                public_id=f"frame_{frame_id}_{anomaly_reason}",
                resource_type="image",
                overwrite=True
            )

            # ✅ Remove temp file immediately after upload
            os.remove(tmp_path)

            return upload_result.get("secure_url")

        except Exception as e:
            print(f"[CloudinaryUploader] Upload failed for frame {frame_id}: {e}")
            traceback.print_exc()
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None
