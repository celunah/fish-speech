from .braceexpand import braceexpand
from .context import autocast_exclude_mps
from .file import get_latest_checkpoint
from .utils import extras, get_metric_value, set_seed, task_wrapper


def __getattr__(name: str):
    if name == "RankedLogger":
        from .logger import RankedLogger

        return RankedLogger

    if name in {"instantiate_callbacks", "instantiate_loggers"}:
        from .instantiators import instantiate_callbacks, instantiate_loggers

        return {
            "instantiate_callbacks": instantiate_callbacks,
            "instantiate_loggers": instantiate_loggers,
        }[name]

    if name == "log_hyperparameters":
        from .logging_utils import log_hyperparameters

        return log_hyperparameters

    if name in {"enforce_tags", "print_config_tree"}:
        from .rich_utils import enforce_tags, print_config_tree

        return {
            "enforce_tags": enforce_tags,
            "print_config_tree": print_config_tree,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "enforce_tags",
    "extras",
    "get_metric_value",
    "RankedLogger",
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "print_config_tree",
    "task_wrapper",
    "braceexpand",
    "get_latest_checkpoint",
    "autocast_exclude_mps",
    "set_seed",
]
