import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from utils.audio_utils import vad_segments, segment_audio, mfcc_embed
from scipy.signal import resample

class AudioDetector:
    """
    Minimal real-time audio detector:
      - Detects multiple speakers
      - Detects general noise presence
    """

    def __init__(self, sr=16000, max_segments=20, cluster_interval=10):
        self.sr = sr
        self.max_segments = max_segments
        self.cluster_interval = cluster_interval
        self._frame_counter = 0

    def detect(self, audio):
        report = {
            "multiple_speakers": False,
            "noise_detected": False  # minimal noise detection
        }

        if audio is None or len(audio) == 0:
            return report

        # Downsample for speed
        if self.sr > 8000:
            target_len = int(len(audio) * 8000 / self.sr)
            audio = resample(audio, target_len)
            self.sr = 8000

        # Voice Activity Detection
        seg_times = vad_segments(audio, self.sr)
        if not seg_times:
            return report

        # If any segment exists → noise detected
        report["noise_detected"] = True

        # Limit segments
        seg_times = seg_times[-self.max_segments:]
        seg_audio = segment_audio(audio, self.sr, seg_times)

        embeddings = []
        for seg in seg_audio:
            try:
                feat = mfcc_embed(seg, self.sr)
                embeddings.append(feat)
            except Exception as e:
                print(f"[AudioDetector] Warning: skipped segment due to error: {e}")

        # Multiple speaker detection via clustering
        if len(embeddings) > 1:
            self._frame_counter += 1
            if self._frame_counter % self.cluster_interval == 0:
                try:
                    X = np.vstack(embeddings)
                    X = StandardScaler().fit_transform(X)

                    n_clusters = min(3, len(embeddings))
                    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=32, n_init=5)
                    labels = kmeans.fit_predict(X)

                    report["multiple_speakers"] = len(set(labels)) > 1
                except Exception as e:
                    print(f"[AudioDetector] Clustering error: {e}")

        return report
