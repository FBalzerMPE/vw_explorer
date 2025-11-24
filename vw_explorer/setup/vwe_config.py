from pathlib import Path
from typing import Dict, Union, Optional
import yaml
from dataclasses import dataclass, field
from .config_io import generate_default_config, sanitize_path, ask_user_confirmation

from ..logger import LOGGER
from .constants import USER_CONFIG_PATH, DEFAULT_CONFIG_PATH



@dataclass
class VWEConfig:
    """Dataclass to manage the VWE configuration."""
    _data: Path = field(init=False, repr=False)
    _output: Path = field(init=False, repr=False)
    _observations: Path = field(init=False, repr=False)
    _guider: Path = field(init=False, repr=False)

    def __post_init__(self):
        # Load the configuration file
        config = self._load_config()
        self._set_paths(config)
    
    def _set_paths(self, config: dict, explicit: bool = False):
        for key, path in config["paths"].items():
            config["paths"][key] = str(sanitize_path(path))
        if explicit:
            self.data_dir = Path(config["paths"]["data_dir"])
            self.output_dir = Path(config["paths"]["output_dir"])
            self.obs_dir = Path(config["paths"]["observation_dir"])
            self.guider_dir = Path(config["paths"]["guider_dir"])
        self._data = Path(config["paths"]["data_dir"])
        self._output = Path(config["paths"]["output_dir"])
        self._observations = Path(config["paths"]["observation_dir"])
        self._guider = Path(config["paths"]["guider_dir"])

    def _load_config(self) -> Dict[str, Dict]:
        """Loads the configuration file."""
        cfg_path = USER_CONFIG_PATH
        if not USER_CONFIG_PATH.exists():
            if ask_user_confirmation(
                f"No user configuration file found at {USER_CONFIG_PATH}.\nWould you like to create a default configuration file there?"
            ):
                generate_default_config()
            else:
                LOGGER.warning("NO USER CONFIGURATION FILE FOUND. USING DEFAULT CONFIG.")
                cfg_path = DEFAULT_CONFIG_PATH

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)


        self._validate_config(cfg)

        return cfg
    
    
    def _validate_config(self, config: dict):
        for key, path in config["paths"].items():
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Path for '{key}' does not exist: {path}\nEither create it, or change it in the config file (vw_explorer/config.yml)."
                )

    def set_to_example_dirs(self):
        """Sets the configuration to the default value."""
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)

        self._set_paths(cfg, explicit=True)
    
    def count_available_files(self) -> Dict[str, int]:
        """Counts the available fits files in the guider and observation directories."""
        avail_dict = {}
        avail_dict["observations"] = len(
            list(self.obs_dir.rglob("vw*.fits"))
        )
        avail_dict["guider frames"] = len(
            list(self.guider_dir.rglob("??????.fits"))
        )
        return avail_dict


    @property
    def data_dir(self) -> Path:
        return self._data

    @data_dir.setter
    def data_dir(self, value: Union[str, Path]):
        self._data = Path(value)
        LOGGER.info(f"Data directory updated to: {self._data}")

    @property
    def output_dir(self) -> Path:
        return self._output

    @output_dir.setter
    def output_dir(self, value: Union[str, Path]):
        self._output = Path(value)
        LOGGER.info(f"Output directory updated to: {self._output}")

    @property
    def obs_dir(self) -> Path:
        return self._observations

    @obs_dir.setter
    def obs_dir(self, value: Union[str, Path]):
        self._observations = Path(value)
        LOGGER.info(f"Observations directory updated to: {self._observations}")

    @property
    def guider_dir(self) -> Path:
        return self._guider

    @guider_dir.setter
    def guider_dir(self, value: Union[str, Path]):
        self._guider = Path(value)
        LOGGER.info(f"Guider directory updated to: {self._guider}")

    def sanitize_logfile_path(self, input_path: Optional[Path]) -> Optional[Path]:
        """Sanitizes the logfile path based on the data directory and input path.
        Will try to use a default log.txt file in the data directory if no input path is provided."""
        logfile_path = self.data_dir if input_path is None else Path(input_path)
        if logfile_path.is_dir():
            logfile_path = logfile_path / "log.txt"
        if not logfile_path.exists():
            return None
        if input_path is None:
            LOGGER.info(f"No log file provided, using default at {logfile_path}")
        return logfile_path
