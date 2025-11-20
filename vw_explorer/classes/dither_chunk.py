from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .observation import Observation
from .observation_sequence import ObservationSequence


@dataclass
class DitherChunk:
    """Represents a chunk of observations for a single target based on dither pattern."""

    obs_seq: ObservationSequence
    chunk_index: int

    def __post_init__(self):
        assert (
            len(self.obs_seq.all_targets) == 1
        ), "DitherChunk must contain observations for a single target."

    def __len__(self) -> int:
        """Returns the number of observations in the chunk."""
        return len(self.obs_seq)

    def __repr__(self) -> str:
        start_time, end_time = self.obs_seq.time_range
        return (
            f"DitherChunk(target={self.target}, "
            f"chunk_id={self.chunk_index}, "
            f"num_obs={len(self)}, "
            f"time_range=({start_time}, {end_time}))"
        )

    @property
    def time_range(self) -> Tuple[datetime, datetime]:
        """Returns the date range of the observations in the chunk."""
        return self.obs_seq.time_range

    @property
    def target(self) -> str:
        """Returns the target name for this dither chunk."""
        return self.obs_seq.all_targets[0]

    @property
    def mean_fiducial_coords(self) -> Tuple[float, float]:
        """Calculates the mean fiducial coordinates across all observations in the chunk."""
        fid_coords = np.array(
            [obs.fiducial_coords for obs in self.obs_seq.observations]
        )
        mean_x, mean_y = np.nanmean(fid_coords, axis=0)
        return mean_x, mean_y
    
    @property
    def is_sky_obs(self) -> bool:
        """Checks if all observations in the chunk are sky observations."""
        return all(obs.is_sky_obs for obs in self.obs_seq.observations)

    @classmethod
    def from_observations(
        cls,
        obs_seq: Union[ObservationSequence, List["Observation"]],
        target_name: str,
        chunk_identifier: int = 0,
    ) -> "DitherChunk":
        """
        Creates a DitherChunk from an ObservationSequence or list of observations.

        Parameters
        ----------
        obs_seq : Union[ObservationSequence, List[Observation]]
            The observation sequence or list of observations to process.
        target_name : str
            The name of the target for which to create the chunk.
        chunk_identifier : int
            The identifier of the chunk to create.

        Returns
        -------
        DitherChunk
            The requested dither chunk.
        """
        chunks = cls.get_dither_chunks_for_target(obs_seq, target_name)

        for chunk in chunks:
            if chunk.chunk_index == chunk_identifier:
                return chunk

        raise ValueError(
            f"No dither chunk found for target '{target_name}' with chunk ID {chunk_identifier}."
        )

    @classmethod
    def from_series(
        cls,
        series: pd.Series,
    ) -> "DitherChunk":
        """
        Creates a DitherChunk from a pandas Series.

        Parameters
        ----------
        series : pd.Series
            A pandas Series containing 'observations' and 'chunk_index'.

        Returns
        -------
        DitherChunk
            The created DitherChunk object.
        """
        observations = series["observation_paths"]
        if not isinstance(observations, list):
            # try to parse as list of filenames:
            observations = [obs.strip() for obs in observations.strip("[]").split(",")]
        if not all(isinstance(obs, str) for obs in observations):
            raise ValueError(
                "Series 'observation_paths' must be a list of Observation objects or filenames."
            )
        chunk_index: int = series["chunk_index"]  # type: ignore
        fid_x, fid_y = series.get("fid_x_mean", float("nan")), series.get(
            "fid_y_mean", float("nan")
        )
        observations = [
            Observation.from_fits(fname, fid_x, fid_y) for fname in observations
        ]
        obs_seq = ObservationSequence(observations=observations)
        return cls(obs_seq=obs_seq, chunk_index=chunk_index)

    @staticmethod
    def get_dither_chunks_for_target(
        obs_seq: Union[ObservationSequence, List["Observation"]], target_name: str
    ) -> List["DitherChunk"]:
        """
        Splits observations for a given target into dither chunks.

        Returns
        -------
        List[DitherChunk]
            A list of DitherChunk objects for the specified target.
        """
        # Filter observations for the target
        target_obs = [obs for obs in obs_seq if obs.target == target_name]
        if not target_obs:
            raise ValueError(f"No observations found for target '{target_name}'.")

        # Group observations into chunks based on dither pattern
        obs_by_chunks = []
        current_obs_chunk = []
        for i, obs in enumerate(target_obs):
            if i == 0 or target_obs[i - 1].dither == obs.dither - 1:
                current_obs_chunk.append(obs)
            else:
                obs_by_chunks.append(current_obs_chunk)
                current_obs_chunk = [obs]

        if current_obs_chunk:
            obs_by_chunks.append(current_obs_chunk)
        d_chunks = [
            DitherChunk(
                obs_seq=ObservationSequence(observations=chunk_obs), chunk_index=i
            )
            for i, chunk_obs in enumerate(obs_by_chunks)
        ]
        return d_chunks

    @staticmethod
    def get_all_dither_chunks(
        observations: Union[ObservationSequence, List["Observation"]],
    ) -> Dict[str, List["DitherChunk"]]:
        """Returns all dither chunks within the observation sequence, optionally for a specific target."""
        chunks = {}
        if not isinstance(observations, ObservationSequence):
            observations = ObservationSequence(observations=observations)
        for target in observations.all_targets:
            target_chunks = DitherChunk.get_dither_chunks_for_target(
                observations, target
            )
            chunks[target] = target_chunks
        return chunks

    @staticmethod
    def to_dataframe(chunks: List["DitherChunk"]) -> pd.DataFrame:
        """
        Creates a DataFrame summarizing dither chunks from a list of observations.
        """
        records = []
        for chunk in chunks:
            fid_coords = chunk.mean_fiducial_coords
            record = {
                "target": chunk.target,
                "observation_names": [obs.filename for obs in chunk.obs_seq],
                "chunk_index": chunk.chunk_index,
                "num_observations": len(chunk),
                "start_time_ut": chunk.time_range[0],
                "end_time_ut": chunk.time_range[1],
                "observation_paths": [obs.fpath for obs in chunk.obs_seq],
                "fid_x_mean": fid_coords[0],
                "fid_y_mean": fid_coords[1],
                "is_sky_obs": chunk.is_sky_obs,
            }
            records.append(record)
        df = pd.DataFrame.from_records(records)
        return df

    def get_summary(self, max_line_length: Optional[int] = None) -> str:
        """Returns a summary string for the dither chunk."""
        return self.obs_seq.get_summary(max_line_length=max_line_length)

    def plot_summary(self):
        """Plots summary statistics for the observation sequence."""
        from ..plotting import plot_guider_sequence_summary

        obs = self.obs_seq
        plot_guider_sequence_summary(obs)
