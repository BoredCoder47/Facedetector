import psycopg2
import os
import time
from dotenv import load_dotenv


class PostgresLogger:
    def __init__(self, db_config=None):
        load_dotenv()

        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            raise EnvironmentError("DATABASE_URL not found.")

        try:
            self.conn = psycopg2.connect(
                DATABASE_URL,
                sslmode="require"
            )
            self.cur = self.conn.cursor()
            self._ensure_tables()
            print("✅ Postgres connected successfully.")
        except Exception as e:
            print("❌ Postgres connection failed:", e)
            raise

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        # Face/video anomalies
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                frame_id INTEGER,
                timestamp TIMESTAMP,
                recognized_name TEXT,
                expected_user TEXT,
                face_visible BOOLEAN,
                eyes_visible BOOLEAN,
                looking_away BOOLEAN,
                multiple_faces BOOLEAN,
                imposter_detected BOOLEAN,
                frame_url TEXT                     -- ✅ NEW: store Cloudinary URL
            );
        """)
        # Simplified audio anomalies table
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS audio_anomalies (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                timestamp TIMESTAMP,
                source TEXT,
                multiple_speakers BOOLEAN,
                noise_detected BOOLEAN
            );
        """)
        self.conn.commit()

    # ----------------- FACE/VIDEO LOGGING -----------------
    def log_anomaly(self, frame_id, name, expected_user,
                    face_visible, eyes_visible, looking_away,
                    multiple_faces, imposter_detected,
                    session_id=None, frame_url=None):
        """
        Logs a single video anomaly event with optional Cloudinary frame URL.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.cur.execute("""
            INSERT INTO anomalies (
                session_id, frame_id, timestamp, recognized_name, expected_user,
                face_visible, eyes_visible, looking_away, multiple_faces, imposter_detected, frame_url
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            session_id, frame_id, timestamp, name, expected_user,
            face_visible, eyes_visible, looking_away,
            multiple_faces, imposter_detected, frame_url
        ))
        self.conn.commit()

    # ----------------- AUDIO LOGGING -----------------
    def log_audio_anomalies(self, session_id, source, multiple_speakers, noise_detected):
        """
        Log minimal audio anomalies (no segment tracking).
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.cur.execute("""
            INSERT INTO audio_anomalies (
                session_id, timestamp, source, multiple_speakers, noise_detected
            )
            VALUES (%s,%s,%s,%s,%s)
        """, (
            session_id, timestamp, source, multiple_speakers, noise_detected
        ))
        self.conn.commit()

    # ----------------- CLEANUP -----------------
    def close(self):
        self.cur.close()
        self.conn.close()
