from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf
import torch


AudioSource = str | Path | bytes | bytearray | BinaryIO


def load_audio_tensor(source: AudioSource) -> tuple[torch.Tensor, int]:
    """Load audio without torchaudio.load/TorchCodec.

    Newer torchaudio versions may route loading through torchcodec DLLs. On
    Windows, mismatched user-site torchcodec wheels can show modal DLL errors,
    so inference uses soundfile directly for stable local loading.
    """

    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    elif hasattr(source, "seek"):
        source.seek(0)

    try:
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load audio with soundfile. Avoiding torchaudio.load because "
            "it may import TorchCodec; pass a WAV/FLAC/OGG file supported by "
            "libsndfile, or fix the installed torch/torchcodec version mismatch."
        ) from exc

    if audio.size == 0:
        raise RuntimeError("Loaded audio is empty.")

    audio = np.ascontiguousarray(audio.T)
    return torch.from_numpy(audio), int(sample_rate)
