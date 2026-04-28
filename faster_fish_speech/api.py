from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal
import weakref

import numpy as np
import torch
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.runtime import (
    RuntimeConfig,
    apply_runtime_config,
    configure_logging,
    get_runtime_config,
)
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest


PrecisionName = Literal["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"]


def _resolve_dtype(dtype: PrecisionName | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def _read_reference_audio(reference_audio: str | Path | bytes | None) -> bytes | None:
    if reference_audio is None:
        return None
    if isinstance(reference_audio, bytes):
        return reference_audio
    return Path(reference_audio).read_bytes()


def _resolve_model_dir(model_path: str | Path) -> Path:
    local_path = Path(model_path).expanduser()
    if local_path.exists():
        return local_path

    repo_id = str(model_path)
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
    except ImportError as exc:
        raise FileNotFoundError(
            f"Model path does not exist locally: {local_path}. "
            "Install the inference dependencies with huggingface-hub support, "
            "or pass a valid local checkpoint directory."
        ) from exc

    try:
        logger.bind(user=True).info(
            f"model path not found locally, trying Hugging Face: {repo_id}"
        )
        return Path(snapshot_download(repo_id=repo_id))
    except (HFValidationError, RepositoryNotFoundError) as exc:
        raise FileNotFoundError(
            f"Model path was not found locally and is not an available "
            f"Hugging Face repo: {model_path!r}"
        ) from exc
    except Exception as exc:
        raise FileNotFoundError(
            f"Model path was not found locally and could not be downloaded "
            f"from Hugging Face: {model_path!r}. Original error: {exc}"
        ) from exc


class S2ProTTS:
    """High-level inference wrapper for faster-fish-speech S2-Pro checkpoints."""

    def __init__(
        self,
        *,
        model_dir: Path,
        engine: TTSInferenceEngine,
        config: RuntimeConfig,
        dtype: torch.dtype,
    ) -> None:
        self.model_dir = model_dir
        self.engine = engine
        self.config = config
        self.dtype = dtype
        self._finalizer = weakref.finalize(self, self._close_engine, engine)

    @staticmethod
    def _close_engine(engine: TTSInferenceEngine) -> None:
        engine.close()

    def close(self) -> None:
        """Release the decoder model, LLaMA worker thread, and CUDA caches."""
        self._finalizer()

    def __enter__(self) -> "S2ProTTS":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        decoder_device: str | torch.device | None = None,
        dtype: PrecisionName | torch.dtype = "bf16",
        optim_level: int | None = None,
        use_cudagraphs: bool | None = None,
        int8_backend: str | None = None,
        compile: bool = False,
        decoder_config: str = "modded_dac_vq",
    ) -> "S2ProTTS":
        config = get_runtime_config(
            optim_level=optim_level,
            use_cudagraphs=use_cudagraphs,
            int8_backend=int8_backend,
        )
        apply_runtime_config(config)
        configure_logging(config)

        model_dir = _resolve_model_dir(model_path)
        device = torch.device(device)
        if decoder_device is None:
            decoder_device = "cpu" if config.optim_level >= 2 and device.type == "cuda" else device
        decoder_device = torch.device(decoder_device)
        precision = _resolve_dtype(dtype)

        llama_queue = launch_thread_safe_queue(
            checkpoint_path=model_dir,
            device=device,
            precision=precision,
            compile=compile,
        )
        decoder_model = load_decoder_model(
            config_name=decoder_config,
            checkpoint_path=model_dir / "codec.pth",
            device=decoder_device,
        )
        engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=compile,
        )
        return cls(model_dir=model_dir, engine=engine, config=config, dtype=precision)

    def _request(
        self,
        *,
        text: str,
        reference_audio: str | Path | bytes | None = None,
        reference_text: str = "",
        streaming: bool,
        **kwargs,
    ) -> ServeTTSRequest:
        audio = _read_reference_audio(reference_audio)
        references = (
            [ServeReferenceAudio(audio=audio, text=reference_text)] if audio else []
        )
        return ServeTTSRequest(
            text=text,
            references=references,
            reference_id=kwargs.pop("reference_id", None),
            streaming=streaming,
            **kwargs,
        )

    def stream(
        self,
        *,
        text: str,
        reference_audio: str | Path | bytes | None = None,
        reference_text: str = "",
        **kwargs,
    ) -> Iterator[tuple[int, np.ndarray]]:
        request = self._request(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            streaming=True,
            **kwargs,
        )
        results = self.engine.inference(request)
        try:
            for result in results:
                if result.error is not None:
                    raise result.error
                if result.code not in {"segment", "chunk"} or result.audio is None:
                    continue
                yield result.audio
        except BaseException:
            results.close()
            raise

    def generate(
        self,
        *,
        text: str,
        reference_audio: str | Path | bytes | None = None,
        reference_text: str = "",
        **kwargs,
    ) -> tuple[int, np.ndarray]:
        request = self._request(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            streaming=False,
            **kwargs,
        )
        results = self.engine.inference(request)
        try:
            for result in results:
                if result.error is not None:
                    raise result.error
                if result.code == "final" and result.audio is not None:
                    return result.audio
        except BaseException:
            results.close()
            raise
        raise RuntimeError("No audio was generated.")
