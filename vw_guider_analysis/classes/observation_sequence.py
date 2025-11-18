from dataclasses import dataclass, field
from typing import List

from .observation import Observation


@dataclass
class ObservationSequence:
    """Represents a sequence of observations for analysis."""

    observations: List[Observation] = field(repr=False)

    def __post_init__(self):
        assert all(
            obs.target == self.observations[0].target for obs in self.observations
        ), "All observations in the sequence should have the same target."

    def __len__(self) -> int:
        return len(self.observations)
