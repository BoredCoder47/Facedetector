import numpy as np
import soundfile as sf
import librosa
import hashlib
import webrtcvad
import sounddevice as sd
from scipy.signal import resample_poly


# ============================================================
# Audio Reading & Preprocessing
# ============================================================
def read_audio(path, target_sr=16000):
    """
    Reads an audio file and returns a mono, resampled numpy array.
    Handles errors gracefully.
    """
    try:
        data, sr = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        data = normalize_audio(data)

        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        return data, sr

    except Exception as e:
        print(f"[AudioUtils] Failed to read {path}: {e}")
        return None, None


def normalize_audio(audio):
    """Clip audio to [-1, 1] to avoid amplitude spikes."""
    if audio is None or len(audio) == 0:
        return np.array([], dtype=np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ============================================================
# Voice Activity Detection (VAD)
# ============================================================
def vad_segments(audio, sr, aggressiveness=2, frame_ms=30, padding_ms=300):
    """
    Detects voice activity regions using WebRTC VAD.
    Returns list of (start_sec, end_sec) tuples.
    """
    if audio is None or len(audio) == 0:
        return []

    v = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * frame_ms / 1000)
    frame_time = frame_ms / 1000.0

    # Ensure frames have correct int16 format
    frames = [audio[i:i + frame_len] for i in range(0, len(audio), frame_len)]
    segments = []
    start = None

    for i, f in enumerate(frames):
        if len(f) < frame_len:
            continue
        try:
            int16_bytes = (f * 32767).astype(np.int16).tobytes()
            is_speech = v.is_speech(int16_bytes, sr)
        except Exception:
            continue

        if is_speech and start is None:
            start = i
        elif not is_speech and start is not None:
            s = max(0, start * frame_time - padding_ms / 1000)
            e = min(len(audio) / sr, i * frame_time + padding_ms / 1000)
            if e > s:
                segments.append((s, e))
            start = None

    if start is not None:
        s = max(0, start * frame_time - padding_ms / 1000)
        e = len(audio) / sr
        segments.append((s, e))

    return segments


# ============================================================
# Segment Extraction
# ============================================================
def segment_audio(audio, sr, segments):
    """
    Converts (start, end) pairs into sliced numpy arrays.
    """
    if audio is None or len(audio) == 0:
        return []

    segs = []
    for s, e in segments:
        a, b = int(s * sr), int(e * sr)
        seg = audio[a:b]
        if len(seg) > 128:  # ignore too-short clips
            segs.append(seg)
    return segs


# ============================================================
# Feature Extraction (MFCC Embedding)
# ============================================================
def mfcc_embed(segment, sr, n_mfcc=13):
    """
    Extract MFCC mean + std features (optimized).
    Downsamples to 8kHz for faster computation if needed.
    """
    if len(segment) < 256:
        segment = np.pad(segment, (0, 256 - len(segment)))

    # Downsample for speed if high SR
    if sr > 8000:
        segment = resample_poly(segment, 1, 2)
        sr = sr // 2

    try:
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        return np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1)))
    except Exception as e:
        print(f"[AudioUtils] MFCC error: {e}")
        return np.zeros(n_mfcc * 2, dtype=np.float32)


# ============================================================
# Segment Hashing
# ============================================================
def md5_of_segment(segment):
    """
    Generates a stable hash for an audio segment.
    """
    if len(segment) == 0:
        return "empty"
    try:
        q = (segment * 32767).astype(np.int16).tobytes()
        return hashlib.md5(q).hexdigest()
    except Exception:
        return "hash_err"


# ============================================================
# Safe Microphone Capture
# ============================================================
def safe_record_audio(duration_sec=2, samplerate=16000):
    """
    Records from microphone safely (returns np.ndarray or None).
    """
    try:
        input_devices = [i for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
        if not input_devices:
            print("[AudioUtils] No microphone detected.")
            return None

        sd.default.device = (input_devices[0], None)
        rec = sd.rec(int(duration_sec * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        return normalize_audio(rec.flatten())

    except Exception as e:
        print(f"[AudioUtils] Mic capture failed: {e}")
        return None
