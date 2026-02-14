import csv
from pathlib import Path

def log_anomalies_csv(segments, session_id, source, multiple_speakers=False,
                      repeated_hashes=None, file_path="audio_anomalies.csv"):
    """
    Log audio anomalies to CSV.

    segments: list of segment dicts from AudioDetector
    multiple_speakers: True if more than 1 speaker detected in the whole audio
    repeated_hashes: list of segment hashes that are repeated
    """
    repeated_hashes = repeated_hashes or []
    if not segments:
        return

    file_exists = Path(file_path).exists()
    with open(file_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "session_id", "source", "anomaly_type", "start_time", "end_time", "speaker_cluster"
        ])
        if not file_exists:
            writer.writeheader()

        for seg in segments:
            anomalies = []

            # Mark multiple speakers
            if multiple_speakers:
                anomalies.append("multiple_speakers")

            # Mark repeated segments
            if seg.get("hash") in repeated_hashes:
                anomalies.append("repeated_audio")

            # Default anomaly if none
            if not anomalies:
                anomalies.append("speech_segment")

            for anomaly_type in anomalies:
                writer.writerow({
                    "session_id": session_id,
                    "source": source,
                    "anomaly_type": anomaly_type,
                    "start_time": seg.get("start"),
                    "end_time": seg.get("end"),
                    "speaker_cluster": seg.get("speaker_cluster")
                })
