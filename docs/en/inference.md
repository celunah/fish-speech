# Inference

The Fish Audio S2 model requires a large amount of VRAM. We recommend using a GPU with at least 24GB for inference.

## Download Weights

First, you need to download the model weights:

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

For a minimal inference install with uv, sync only the inference dependency
group and the CUDA extra that matches your machine:

```bash
uv sync --only-group inference --extra cu129
```

## Runtime Controls

`FASTER_FISH_SPEECH_OPTIM_LEVEL` controls the public optimization profile.
The default is `2`.

| Level | Intent |
| --- | --- |
| `0` | Full precision / no optimization target, roughly 24 GB VRAM |
| `1` | Light optimization target, roughly 16 GB VRAM |
| `2` | Balanced optimization target, roughly 12 GB VRAM |
| `3` | Maximum optimization target, roughly 5.9 GB VRAM with the hybrid INT8 checkpoint |

`FASTER_FISH_SPEECH_USE_CUDAGRAPHS=1` enables optional CUDA graph replay for
the fixed-shape text-to-semantic token generation hot path. It is disabled by
default, only runs on CUDA devices, and falls back to eager mode if capture
fails. Model loading, reference encoding, file I/O, sampling decisions, and the
codec path are not captured.

`FASTER_FISH_SPEECH_INT8_BACKEND` controls optional compiler-free INT8 Linear
backends. The default `auto` tries installed TorchAO or bitsandbytes support
where usable, then falls back to the built-in hybrid INT8/BF16 implementation.
Use `fallback` to force the built-in path. TorchAO and bitsandbytes are optional
and are not required for Windows compatibility.

`FASTER_FISH_SPEECH_INT8_DEQUANT_CACHE_GB` overrides the fallback INT8
dequantized-weight cache size in GB. This can trade VRAM for speed without
changing output quality.

Set `FASTER_FISH_SPEECH_VERBOSE=1` for detailed internal logs, or
`FASTER_FISH_SPEECH_DEV_MODE=1` to show prompt visualization and development
diagnostics.

## Benchmarking

The benchmark helper compares optimization levels and CUDA graph modes while
reporting tokens/sec, latency, VRAM, RAM, and load time:

```bash
python tools/benchmark_faster_inference.py \
    --checkpoint-path checkpoints/s2-pro \
    --text "Text to benchmark" \
    --max-new-tokens 256 \
    --optim-levels 1,2,3 \
    --cudagraphs 0,1
```

For cloned-voice quality checks, include the same reference text and VQ prompt
tokens you use in production:

```bash
python tools/benchmark_faster_inference.py \
    --checkpoint-path checkpoints/s2-pro \
    --text "Text to benchmark" \
    --prompt-text "Reference transcript" \
    --prompt-tokens fake.npy
```

## Quiet Python Inference

Use the high-level `S2ProTTS` wrapper for new scripts. It hides the queue and
decoder setup, keeps the backend quiet by default, and supports CPU decoder
offload for smaller GPUs.

```python
import wave

import numpy as np

from faster_fish_speech import S2ProTTS

tts = S2ProTTS.from_pretrained(
    "lunahr/fishaudio-s2-pro-hybrid-int8",
    device="cuda",
    decoder_device="cpu",
    dtype="bf16",
    optim_level=2,
    use_cudagraphs=True,
    int8_backend="auto",
)

wav = wave.open("output.wav", "wb")
try:
    wrote_header = False

    for sample_rate, audio in tts.stream(
        text="Text to say here",
        reference_audio="path/to/ref",
        reference_text="Audio transcript here",
        max_new_tokens=1024,
        chunk_length=200,
    ):
        if not wrote_header:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wrote_header = True

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)
        wav.writeframes(audio.tobytes())
        wav._file.flush()
finally:
    if wrote_header:
        wav.close()
    else:
        wav._file.close()
```

If `model_dir` does not exist locally, `from_pretrained` treats the value as a
Hugging Face repo id and tries to download it. For example:

```python
tts = S2ProTTS.from_pretrained("fishaudio/s2-pro", device="cuda")
```

If neither the local path nor the Hugging Face repo exists, loading fails with a
clear `FileNotFoundError`.

To fully unload the model in a long-running Python process, close the wrapper
before clearing CUDA caches:

```python
tts.close()
del tts
torch.cuda.empty_cache()
gc.collect()
```

You can also use it as a context manager:

```python
with S2ProTTS.from_pretrained("lunahr/fishaudio-s2-pro-hybrid-int8") as tts:
    sample_rate, audio = tts.generate(text="Text to say here")
```

## Command Line Inference

!!! note
    If you plan to let the model randomly choose a voice timbre, you can skip this step.

### 1. Get VQ tokens from reference audio

```bash
python -m faster_fish_speech.models.dac.inference \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

You should get a `fake.npy` and a `fake.wav`.

### 2. Generate Semantic tokens from text:

```bash
python -m faster_fish_speech.models.text2semantic.inference \
    --text "The text you want to convert" \
    --prompt-text "Your reference text" \
    --prompt-tokens "fake.npy" \
    # --compile
```

This command will create a `codes_N` file in the working directory, where N is an integer starting from 0.

!!! note
    You may want to use `--compile` to fuse CUDA kernels for faster inference. However, we recommend using our sglang inference acceleration optimization.
    Correspondingly, if you do not plan to use acceleration, you can comment out the `--compile` parameter.

!!! info
    For GPUs that do not support bf16, you may need to use the `--half` parameter.

### 3. Generate vocals from semantic tokens:

```bash
python -m faster_fish_speech.models.dac.inference \
    -i "codes_0.npy" \
```

After that, you will get a `fake.wav` file.

## WebUI Inference

### 1. Gradio WebUI

For compatibility, we still maintain the Gradio WebUI.

```bash
python tools/run_webui.py # --compile if you need acceleration
```

### 2. Awesome WebUI

Awesome WebUI is a modernized Web interface built with TypeScript, offering richer features and a better user experience.

**Build WebUI:**

You need to have Node.js and npm installed on your local machine or server.

1. Enter the `awesome_webui` directory:
   ```bash
   cd awesome_webui
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Build the WebUI:
   ```bash
   npm run build
   ```

**Start Backend Server:**

After building the WebUI, return to the project root and start the API server:

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**Access:**

Once the server is running, you can access it via your browser:
`http://localhost:8888/ui`
