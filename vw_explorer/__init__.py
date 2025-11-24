from .calculations import *
from .classes import (
    DitherChunk,
    GuiderFrame,
    GuiderSequence,
    GuideStarModel,
    Observation,
    ObservationSequence,
    ObsTimeslot,
)
from .constants import ASSET_PATH, CONFIG
from .io import *
from .logger import LOGGER
from .plotting import *

_mpstyle_path = ASSET_PATH / "inward_ticks.mplstyle"
if (_mpstyle_path).exists():
    import matplotlib.pyplot as plt

    LOGGER.debug(f"Applying matplotlib style from {str(_mpstyle_path)}")
    plt.style.use(str(_mpstyle_path))
