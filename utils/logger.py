import logging
import sys

def setup_logger():
    """Sets up the global logger."""
    logger = logging.getLogger("IBKR_BOT")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Create a stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(stream_handler)
    return logger

log = setup_logger()