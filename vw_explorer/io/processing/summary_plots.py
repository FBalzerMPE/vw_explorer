from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from ...io import load_observations

from ...classes import GuiderSequence, DitherChunk
from ...logger import LOGGER


def _plot_guider_sequence(gseq: GuiderSequence, output_path: Path):
    """
    Generates and saves plots for a single guider sequence.

    Parameters
    ----------
    gseq : GuiderSequence
        GuiderSequence object.
    output_path : Path
        Path to save the plot.
    """
    try:
        fig = gseq.plot_summary()
        fig.suptitle(f"{gseq.observation.long_name}", fontsize=16)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.debug(f"Saved guider sequence plot to {output_path}")
    except Exception as e:
        LOGGER.warning(f"Error generating guider sequence plot: {e}")


def _plot_dither_chunk_summary(chunk: DitherChunk, output_path: Path):
    """
    Generates and saves a summary plot for a dither chunk.

    Parameters
    ----------
    chunk : DitherChunk
        DitherChunk object.
    output_path : Path
        Path to save the plot.
    """
    try:
        chunk.plot_summary()
        fig = plt.gcf()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.debug(f"Saved dither chunk summary plot to {output_path}")
    except Exception as e:
        LOGGER.warning(f"Error generating dither chunk summary plot: {e}")

def generate_dither_chunk_plots(
    output_dir: Path,
):
    """
    Generates and saves plots for each dither chunk.

    Parameters
    ----------
    dither_chunks : List[DitherChunk]
        List of DitherChunk objects.
    output_dir : Path
        Directory to save the plots.
    """
    observations = load_observations()
    ch_dict = DitherChunk.get_all_dither_chunks(observations)
    dither_chunks = [ch for ch_list in ch_dict.values() for ch in ch_list]

    plot_dir = output_dir / "plots"
    obs_plot_dir = plot_dir / "observations"
    obs_plot_dir.mkdir(parents=True, exist_ok=True)
    dither_chunk_dir = plot_dir / "dither_chunks"
    dither_chunk_dir.mkdir(parents=True, exist_ok=True)
    
    for chunk in dither_chunks:
        if not chunk.is_sky_obs:
            continue
        for gseq in chunk.obs_seq.get_guider_sequences():
            plot_path = obs_plot_dir / f"{gseq.observation.filename}_summary.png"
            _plot_guider_sequence(gseq, plot_path)
        chunk_plot_path = dither_chunk_dir / f"dither_chunk_{chunk.target}_{chunk.chunk_index}_summary.png"
        _plot_dither_chunk_summary(chunk, chunk_plot_path)

