from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from ..constants import CONFIG


@dataclass
class ObsTimeslot:
    """Represents an observation time slot with start and end times."""

    start_time: datetime
    end_time: datetime

    @classmethod
    def from_entry(cls, entry: dict) -> "ObsTimeslot":
        start = entry["UT"]
        start_time = datetime.combine(datetime.today(), start)
        end_time = start_time + timedelta(seconds=int(entry["exptime"]))
        return cls(start_time=start_time, end_time=end_time)

    @classmethod
    def from_start_and_time(cls, start_time: datetime, exptime: float) -> "ObsTimeslot":
        end_time = start_time + timedelta(seconds=int(exptime))
        return cls(start_time=start_time, end_time=end_time)

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    @property
    def mid_time(self) -> datetime:
        return self.start_time + self.duration / 2

    @property
    def summary(self) -> str:
        if self.start_time.date() == self.end_time.date():
            return f"{self.start_time.strftime('%Y-%m-%d %H:%M:%S')} to {self.end_time.strftime('%H:%M:%S')} (duration: {self.duration})"
        return f"{self.start_time} to {self.end_time} (duration: {self.duration})"

    def contains(self, timestamp: datetime) -> bool:
        return self.start_time <= timestamp <= self.end_time

    def load_guider_frames(self) -> list:
        """Returns guider frames that fall within this time slot."""
        # import here to avoid circular imports
        from .guider_frame import GuiderFrame

        guider_index_df = GuiderFrame.get_guider_index()
        mask = guider_index_df["datetime"].apply(self.contains)
        fnames = guider_index_df[mask]["fname"]
        return [GuiderFrame(f) for f in fnames]
