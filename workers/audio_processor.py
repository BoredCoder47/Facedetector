# workers/audio_processor.py
from utils.audio_utils import safe_record_audio
from detectors.audio_detector import AudioDetector
from utils.postgres_logger import PostgresLogger
from utils.session_utils import generate_session_id

class AudioProcessor:
    def __init__(self, db_config, session_id=None, sr=16000):
        self.session_id = session_id or generate_session_id(prefix="audio")
        self.sr = sr
        self.detector = AudioDetector(sr=self.sr)
        self.logger = PostgresLogger(db_config)

    # -----------------------------
    # Process live microphone input
    # -----------------------------
    def process_mic(self, duration_sec=3):
        audio = safe_record_audio(duration_sec=duration_sec, samplerate=self.sr)
        if audio is None or len(audio) == 0:
            print("[AudioProcessor] No audio captured.")
            return {"multiple_speakers": False, "noise_detected": False}

        if audio.ndim > 1:
            audio = audio.flatten()

        # Detect anomalies (minimal)
        report = self.detector.detect(audio)
        multiple_speakers = report.get("multiple_speakers", False)
        noise_detected = bool(len(audio) > 0 and not report.get("multiple_speakers", False))  # simple noise flag

        # Log minimal info
        try:
            self.logger.log_audio_anomalies(
                segments=None,  # not tracking segments
                session_id=self.session_id,
                source="microphone",
                multiple_speakers=multiple_speakers,
                noise_detected=noise_detected
            )
        except Exception as e:
            print(f"[AudioProcessor] Logging error: {e}")

        return {"multiple_speakers": multiple_speakers, "noise_detected": noise_detected}

    # -----------------------------
    # Cleanup
    # -----------------------------
    def close(self):
        self.logger.close()
