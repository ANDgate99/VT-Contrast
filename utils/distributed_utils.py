import logging
import os

logger = logging.getLogger(__name__)


def get_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r. Falling back to %s.", name, raw_value, default)
        return default


def get_rank() -> int:
    return get_env_int("RANK", 0)


def get_world_size() -> int:
    return max(get_env_int("WORLD_SIZE", 1), 1)


def is_main_process() -> bool:
    return get_rank() == 0
