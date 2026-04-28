import ctypes
import os
import sys
import warnings
from dataclasses import dataclass

from loguru import logger


OPTIM_LEVEL_ENV = "FASTER_FISH_SPEECH_OPTIM_LEVEL"
CUDA_GRAPHS_ENV = "FASTER_FISH_SPEECH_USE_CUDAGRAPHS"
INT8_BACKEND_ENV = "FASTER_FISH_SPEECH_INT8_BACKEND"
INT8_DEQUANT_CACHE_GB_ENV = "FASTER_FISH_SPEECH_INT8_DEQUANT_CACHE_GB"
DEV_MODE_ENV = "FASTER_FISH_SPEECH_DEV_MODE"
VERBOSE_ENV = "FASTER_FISH_SPEECH_VERBOSE"
PROMPT_VISUALIZE_ENV = "FASTER_FISH_SPEECH_DISABLE_PROMPT_VISUALIZE"


@dataclass(frozen=True)
class RuntimeConfig:
    optim_level: int = 2
    use_cudagraphs: bool = False
    int8_backend: str = "auto"
    dev_mode: bool = False
    verbose: bool = False

    @property
    def label(self) -> str:
        return f"O{self.optim_level}"


def _parse_bool(value: str | bool | int | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_optim_level(value: str | int | None, default: int = 2) -> int:
    if value is None:
        return default
    try:
        level = int(value)
    except (TypeError, ValueError):
        return default
    return min(3, max(0, level))


def get_runtime_config(
    *,
    optim_level: int | None = None,
    use_cudagraphs: bool | None = None,
    int8_backend: str | None = None,
    dev_mode: bool | None = None,
    verbose: bool | None = None,
) -> RuntimeConfig:
    backend = (int8_backend or os.environ.get(INT8_BACKEND_ENV) or "auto").lower()
    if backend not in {"auto", "torchao", "bitsandbytes", "bnb", "fallback"}:
        backend = "auto"
    return RuntimeConfig(
        optim_level=_parse_optim_level(
            optim_level if optim_level is not None else os.environ.get(OPTIM_LEVEL_ENV)
        ),
        use_cudagraphs=_parse_bool(
            use_cudagraphs
            if use_cudagraphs is not None
            else os.environ.get(CUDA_GRAPHS_ENV),
            default=False,
        ),
        int8_backend="bitsandbytes" if backend == "bnb" else backend,
        dev_mode=_parse_bool(
            dev_mode if dev_mode is not None else os.environ.get(DEV_MODE_ENV),
            default=False,
        ),
        verbose=_parse_bool(
            verbose if verbose is not None else os.environ.get(VERBOSE_ENV),
            default=False,
        ),
    )


def apply_runtime_config(config: RuntimeConfig) -> None:
    os.environ[OPTIM_LEVEL_ENV] = str(config.optim_level)
    os.environ[CUDA_GRAPHS_ENV] = "1" if config.use_cudagraphs else "0"
    os.environ[INT8_BACKEND_ENV] = config.int8_backend
    os.environ[DEV_MODE_ENV] = "1" if config.dev_mode else "0"
    os.environ[VERBOSE_ENV] = "1" if config.verbose else "0"


def configure_logging(config: RuntimeConfig | None = None) -> None:
    configure_warning_filters()
    config = config or get_runtime_config()
    logger.remove()

    def log_filter(record):
        if config.verbose or config.dev_mode:
            return True
        if record["level"].no >= logger.level("WARNING").no:
            return True
        return bool(record["extra"].get("user"))

    logger.add(sys.stderr, level="INFO", filter=log_filter, format="{message}")


def configure_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.nn\.utils\.weight_norm.*deprecated in favor of.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Config Deprecation: version 1 of Int8WeightOnlyConfig.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Deprecation: (PlainLayout|PlainAQTTensorImpl|AffineQuantizedTensor).*",
        category=UserWarning,
    )


def should_visualize_prompts() -> bool:
    if _parse_bool(os.environ.get(PROMPT_VISUALIZE_ENV), default=False):
        return False
    return get_runtime_config().dev_mode


def int8_dequant_cache_limit_bytes(config: RuntimeConfig | None = None) -> int:
    override = os.environ.get(INT8_DEQUANT_CACHE_GB_ENV)
    if override is not None:
        try:
            return max(0, int(float(override) * (1024**3)))
        except ValueError:
            pass

    config = config or get_runtime_config()
    limits_gb = {
        0: 24.0,
        1: 8.0,
        2: 2.0,
        3: 0.0,
    }
    return int(limits_gb[config.optim_level] * (1024**3))


def system_ram_used_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().used / 1e9
    except Exception:
        pass

    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return (status.ullTotalPhys - status.ullAvailPhys) / 1e9


def log_benchmark(
    *,
    tokens_sec: float,
    latency_sec: float | None = None,
    vram_gb: float | None,
    config: RuntimeConfig | None = None,
) -> None:
    config = config or get_runtime_config()
    ram_gb = system_ram_used_gb()
    ram_text = f"{ram_gb:.2f} GB" if ram_gb is not None else "unknown"
    vram_text = f"{vram_gb:.2f} GB" if vram_gb is not None else "n/a"
    latency_text = f"{latency_sec:.2f}s" if latency_sec is not None else "n/a"
    logger.bind(user=True).info(
        "benchmark: "
        f"{tokens_sec:.2f} tokens/sec | "
        f"latency {latency_text} | "
        f"VRAM {vram_text} | "
        f"RAM {ram_text} | "
        f"optim {config.label} | "
        f"cudagraphs {'on' if config.use_cudagraphs else 'off'} | "
        f"int8 {config.int8_backend}"
    )
