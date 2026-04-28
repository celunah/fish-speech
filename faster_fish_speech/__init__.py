"""Compatibility import package for faster-fish-speech.

The implementation still lives under ``fish_speech`` internally so existing
scripts keep working, while new code can import ``faster_fish_speech``.
"""

from importlib import import_module

_own_path = list(__path__)

from faster_fish_speech.api import S2ProTTS
from fish_speech.runtime import RuntimeConfig, apply_runtime_config, get_runtime_config

_impl = import_module("fish_speech")
__path__ = list(_impl.__path__) + _own_path

__all__ = [
    "RuntimeConfig",
    "S2ProTTS",
    "apply_runtime_config",
    "get_runtime_config",
]
