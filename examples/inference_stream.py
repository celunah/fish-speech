from pathlib import Path
import wave

import numpy as np

from faster_fish_speech import S2ProTTS


tts = S2ProTTS.from_pretrained(
    Path("path/to/model"),
    device="cuda",
    decoder_device="cpu",
    dtype="bf16",
    optim_level=3,
)

wav = wave.open("output.wav", "wb")
try:
    wrote_header = False
    for sample_rate, audio in tts.stream(
        text="Hello from faster-fish-speech.",
        reference_audio="refs/calm.wav",
        reference_text="My name is Celune. It is quiet.",
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
        wav.writeframes((audio * 32767.0).astype(np.int16).tobytes())
finally:
    if wrote_header:
        wav.close()
    else:
        wav._file.close()
