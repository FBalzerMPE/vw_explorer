import textwrap
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from ..calculations import get_target_counts
from ..logger import LOGGER
from .guider_sequence import GuiderSequence
from .observation import Observation


@dataclass
class ObservationSequence:
    """Represents a sequence of observations for analysis."""

    observations: List[Observation] = field(repr=False)
    sci_targets: List[str] = field(init=False)
    all_targets: List[str] = field(init=False)
    _guider_sequences: Optional[List[GuiderSequence]] = field(default=None, repr=False)

    def __post_init__(self):
        self.sci_targets = sorted(
            set(obs.target for obs in self.observations if not obs.is_calibration_obs)
        )
        self.all_targets = sorted(set(obs.target for obs in self.observations))
        self.observations.sort(key=lambda x: x.start_time_ut)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> Observation:
        """Allows indexing to access Observation objects."""
        return self.observations[index]

    def __iter__(self):
        """Allows iteration over Observation objects."""
        return iter(self.observations)

    def __repr__(self) -> str:
        cts = get_target_counts(self.observations, remove_calib=False)
        cts_str = ", ".join(f"{k}: {v}" for k, v in cts.items())
        return f"ObservationSequence(num_observations={len(self.observations)}, target_counts={{ {cts_str} }})"

    def __str__(self) -> str:
        return self.get_summary()

    @classmethod
    def from_filenames(cls, filenames: List[Path]) -> "ObservationSequence":
        """Creates an ObservationSequence from a list of filenames."""
        observations = []
        for fname in filenames:
            try:
                obs = Observation.from_fits(fname)
                observations.append(obs)
            except ValueError as e:
                LOGGER.warning(f"Skipping file '{fname}' due to error: {e}")
        assert observations, "No valid observations found from the provided filenames."
        return cls(observations=observations)

    @classmethod
    def from_target(
        cls,
        target: str,
        observations: List[Observation],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> "ObservationSequence":
        """Creates an ObservationSequence for a specific target."""
        target_obs = [obs for obs in observations if obs.target == target]
        assert target_obs, f"No observations found for target '{target}'."
        if start is not None:
            target_obs = [obs for obs in target_obs if obs.start_time_ut >= start]
        if end is not None:
            target_obs = [obs for obs in target_obs if obs.start_time_ut <= end]
        assert (
            target_obs
        ), f"No observations found for target '{target}' in the specified time range."
        return cls(observations=target_obs)

    def _load_guider_sequences(
        self, reload: bool = False, remove_failed: bool = False
    ) -> None:
        """Loads GuiderSequence objects for each observation in the sequence."""
        if self._guider_sequences is not None and not reload:
            if len(self._guider_sequences) == len(self.observations):
                return
            if not remove_failed:
                LOGGER.warning(
                    "Guider sequences already loaded but length mismatch with observations."
                )
                return
        if len(self) > 30:
            LOGGER.warning(
                f"Loading guider sequences for {len(self)} observations may take some time."
            )
        self._guider_sequences = []
        remove_indices = []
        for i, obs in enumerate(self.observations):
            try:
                gs = GuiderSequence(obs)
                self._guider_sequences.append(gs)
            except ValueError as e:
                LOGGER.warning(
                    f"Skipping observation {i} ({obs.filename}) due to error: {e}"
                )
                if not remove_failed:
                    remove_indices.append(i)
        if remove_failed and remove_indices:
            for index in sorted(remove_indices, reverse=True):
                del self.observations[index]

    def get_guider_sequences(
        self, reload: bool = False, remove_failed: bool = True
    ) -> List[GuiderSequence]:
        """Returns the list of GuiderSequence objects for the observations.
        If all observations are calibration frames, returns an empty list.
        """
        if all(obs.is_calibration_obs for obs in self):
            return []
        if self._guider_sequences is None or reload:
            self._load_guider_sequences(reload=reload, remove_failed=remove_failed)
        return self._guider_sequences  # type: ignore

    @property
    def is_single_target(self) -> bool:
        """Returns True if the sequence contains observations for a single target."""
        return len(self.sci_targets) == 1

    @property
    def is_single_dither_chunk(self) -> bool:
        """Returns True if the sequence contains observations forming a single dither chunk for its target."""
        if not self.is_single_target:
            return False
        dither_values = [obs.dither for obs in self.observations]
        # assert whether the differences between consecutive dithers are all 1
        return all(
            (dither_values[i] - dither_values[i - 1] == 1)
            for i in range(1, len(dither_values))
        )

    @property
    def time_range(self) -> Tuple[datetime, datetime]:
        """Returns the start and end time of the chunk."""
        start_time = min(obs.start_time_ut for obs in self)
        end_time = max(obs.start_time_ut for obs in self)
        return start_time, end_time

    def get_summary(self, max_line_length: Optional[int] = None) -> str:
        """Provide a summary string for a list of observations."""
        tr = self.time_range
        earliest = tr[0].isoformat(" ", "seconds")
        latest = tr[1].isoformat(" ", "seconds")
        num_obs = len(self)
        target_counts = Counter(obs.target for obs in self)
        science_targets = {
            k: v for k, v in target_counts.items() if k in self.sci_targets
        }
        science_targets = dict(
            sorted(science_targets.items(), key=lambda item: item[0])
        )
        calib_targets = {
            k: v for k, v in target_counts.items() if k not in self.sci_targets
        }

        summary = "Summary:\n"
        if len(self.sci_targets) == 1:
            summary += f"  Target: {self.sci_targets[0]}\n"
        else:
            summary += (
                f"  Targets ({len(self.sci_targets)}): {', '.join(self.sci_targets)}\n"
            )
            for target, count in science_targets.items():
                summary += f"    {target + ':':<10} {count}\n"
        summary += (            
            f"  Time Range:\n    {earliest} to\n    {latest}\n"
            f"  Total Observations: {num_obs}\n"
        )
        num_avail = sum(obs.file_available for obs in self)
        num_missing = num_obs - num_avail
        summary += f"  Number of Available Files: {num_avail}\n"
        if num_missing > 0:
            summary += f"  Number of Missing Files: {num_missing}\n"
        num_calibs = sum(calib_targets.values())
        if num_calibs > 0:
            summary += f"  Calibration Observations: {num_calibs}\n"
        if len(self) <= 6:
            summary += (
                "  Comments:\n"
                + ";\n".join(
                    f"[D{obs.dither}] {obs.trimmed_comments}"
                    for obs in self
                    if obs.comments
                )
                + "\n"
            )

        # Wrap lines longer than max_line_length
        if max_line_length is None:
            return summary
        wrapped_lines = []
        for line in summary.splitlines():
            if len(line) > max_line_length:
                wrapped_lines.extend(textwrap.wrap(line, width=max_line_length))
            else:
                wrapped_lines.append(line)
        summary = "\n".join(wrapped_lines)
        return summary
