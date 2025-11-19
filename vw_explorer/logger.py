import logging

LOGGER = logging.getLogger("vw-explorer")
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    "[%(asctime)s: %(name)s %(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
_handler.setFormatter(_formatter)
LOGGER.handlers.clear()
LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
