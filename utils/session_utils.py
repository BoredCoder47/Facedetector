# utils/session_utils.py
import uuid
import time

def generate_session_id(prefix="session"):
    """Generate a unique session ID like session_20251024_6f3b2b9c"""
    timestamp = time.strftime("%Y%m%d")
    random_id = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{random_id}"
