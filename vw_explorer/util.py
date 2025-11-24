from datetime import datetime

from .constants import ASSET_PATH

def parse_isoformat(dt_str: str) -> datetime:
    """Parses an ISO 8601 datetime string, handling both with and without microseconds.
    Covers the case of outdated datetime versions that do not implement this natively.
    """
    try:
        return datetime.fromisoformat(dt_str)
    except AttributeError:
        dt_str = dt_str.strip().replace(" ", "T").replace("-", "").replace(":", "")
        if "." in dt_str:
            return datetime.strptime(dt_str, "%Y%m%dT%H%M%S.%f")
        return datetime.strptime(dt_str, "%Y%m%dT%H%M%S")


def try_play_notification_sound():
    soundpath = ASSET_PATH / "notify.wav"
    try:
        if not soundpath.exists():
            return
        from playsound import playsound
        playsound(str(soundpath))
    except ImportError:
        pass
