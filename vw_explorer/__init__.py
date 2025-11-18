from .calculations import *
from .classes import (
    GuiderFrame,
    GuiderSequence,
    GuideStarModel,
    Observation,
    ObservationSequence,
    ObsTimeslot,
)
from .constants import ASSET_PATH, DATA_PATH
from .io import *
from .io.guider_indexing import *
from .logger import LOGGER
from .plotting import *

_mpstyle_path = ASSET_PATH / "inward_ticks.mplstyle"
if (_mpstyle_path).exists():
    import matplotlib.pyplot as plt

    LOGGER.debug(f"Applying matplotlib style from {str(_mpstyle_path)}")
    plt.style.use(str(_mpstyle_path))
