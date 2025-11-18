from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from ..calculations import get_target_counts
from ..constants import CALIB_NAMES
from ..logger import LOGGER
from .guider_sequence import GuiderSequence
from .observation import Observation


@dataclass
class ObservationSequence:
    """Represents a sequence of observations for analysis."""

    observations: List[Observation] = field(repr=False)
    targets: list[str] = field(init=False)
    guider_sequences: Optional[List[GuiderSequence]] = field(default=None, repr=False)

    def __post_init__(self):
        self.targets = sorted(
            set(
                obs.target for obs in self.observations if obs.target not in CALIB_NAMES
            )
        )
        self.observations.sort(key=lambda x: x.start_time_ut_noted)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> Observation:
        """Allows indexing to access Observation objects."""
        return self.observations[index]

    def __repr__(self) -> str:
        cts = get_target_counts(self.observations, remove_calib=False)
        cts_str = ", ".join(f"{k}: {v}" for k, v in cts.items())
        return f"ObservationSequence(num_observations={len(self.observations)}, target_counts={{ {cts_str} }})"

    def __str__(self) -> str:
        """Provide a summary string for a list of observations."""
        obs = self.observations
        earliest, latest = (
            min(obs.start_time_ut_noted for obs in obs),
            max(obs.start_time_ut_noted for obs in obs),
        )
        num_obs = len(obs)
        target_counts = Counter(obs.target for obs in obs)
        science_targets = {
            k: v for k, v in target_counts.items() if k not in CALIB_NAMES
        }
        science_targets = dict(
            sorted(science_targets.items(), key=lambda item: item[0])
        )
        calib_targets = {k: v for k, v in target_counts.items() if k in CALIB_NAMES}

        summary = (
            f"Observation Log Summary:\n"
            f"  Time Range:\n    {earliest} to\n    {latest}\n"
            f"  Total Observations: {num_obs}\n"
        )
        num_avail = sum(obs.file_available for obs in obs)
        num_missing = num_obs - num_avail
        summary += f"  Number of Available Files: {num_avail}\n"
        if num_missing > 0:
            summary += f"  Number of Missing Files: {num_missing}\n"
        if len(self.targets) == 1:
            summary += f"  Target: {self.targets[0]}\n"
        else:
            summary += f"  Targets ({len(self.targets)}): {', '.join(self.targets)}\n"
            for target, count in science_targets.items():
                if target in CALIB_NAMES:
                    continue
                summary += f"    {target + ':':<10} {count}\n"
        num_calibs = sum(calib_targets.values())
        if num_calibs > 0:
            summary += f"  Calibration Observations: {num_calibs}\n"
        return summary

    @property
    def is_single_target(self) -> bool:
        """Returns True if the sequence contains observations for a single target."""
        return len(self.targets) == 1

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
            target_obs = [obs for obs in target_obs if obs.start_time_ut_noted >= start]
        if end is not None:
            target_obs = [obs for obs in target_obs if obs.start_time_ut_noted <= end]
        assert (
            target_obs
        ), f"No observations found for target '{target}' in the specified time range."
        return cls(observations=target_obs)

    def get_dither_chunk(
        self, chunk_index: int = 0, target_name: Optional[str] = None
    ) -> "ObservationSequence":
        """Returns a chunk of observations for a given target based on dither pattern."""
        if target_name is None:
            assert (
                len(self.targets) == 1
            ), "Multiple targets found; please specify a target_name."
            target_name = self.targets[0]
        target_obs = [obs for obs in self.observations if obs.target == target_name]
        assert target_obs, f"No observations found for target '{target_name}'."
        # Whenever we have an observation that is not part of the dither pattern, we start a new chunk
        chunks = []
        current_chunk = []
        for i, obs in enumerate(target_obs):
            if i == 0:
                current_chunk.append(obs)
                continue
            last_obs = target_obs[i - 1]
            if last_obs.dither == obs.dither - 1:
                current_chunk.append(obs)
                continue
            chunks.append(current_chunk)
            current_chunk = [obs]
        if current_chunk:
            chunks.append(current_chunk)
        assert (
            chunks
        ), f"No observations found for target '{target_name}' in the specified dither chunk."
        LOGGER.info(f"Found {len(chunks)} dither chunks for target '{target_name}'.")
        return ObservationSequence(observations=chunks[chunk_index])

    def load_guider_sequences(self, reload: bool = False, remove_failed: bool = False):
        """Loads GuiderSequence objects for each observation in the sequence."""
        if self.guider_sequences is not None and not reload:
            assert len(self.guider_sequences) == len(self.observations)
            return
        if len(self) > 50:
            LOGGER.warning(
                f"Loading guider sequences for {len(self)} observations may take some time."
            )
        self.guider_sequences = []
        remove_indices = []
        for i, obs in enumerate(self.observations):
            try:
                gs = GuiderSequence(obs)
                self.guider_sequences.append(gs)
            except ValueError as e:
                LOGGER.warning(
                    f"Skipping observation {i} ({obs.filename}) due to error: {e}"
                )
                if not remove_failed:
                    remove_indices.append(i)
        if remove_failed and remove_indices:
            for index in sorted(remove_indices, reverse=True):
                del self.observations[index]

    def plot_summary(self):
        """Plots summary statistics for the observation sequence."""
        from ..plotting import plot_guider_sequence_summary

        assert (
            self.is_single_target
        ), "Should only plot summary for single-target sequences."

        if self.guider_sequences is None:
            self.load_guider_sequences()
