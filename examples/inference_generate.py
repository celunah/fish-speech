from pathlib import Path

import soundfile as sf

from faster_fish_speech import S2ProTTS


tts = S2ProTTS.from_pretrained(
    Path("path/to/model"),
    device="cuda",
    decoder_device="cpu",
    dtype="bf16",
    optim_level=2,
)

sample_rate, audio = tts.generate(
    text="Hello from faster-fish-speech.",
    reference_audio="refs/calm.wav",
    reference_text="My name is Celune. It is quiet.",
    max_new_tokens=1024,
)

sf.write("output.wav", audio, sample_rate)
