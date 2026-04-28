from __future__ import annotations

import os
import time
from pathlib import Path

import click
import numpy as np
import torch

from fish_speech.models.text2semantic.inference import generate_long, init_model
from fish_speech.runtime import (
    apply_runtime_config,
    get_runtime_config,
    system_ram_used_gb,
)


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_bools(value: str) -> list[bool]:
    parsed = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if not normalized:
            continue
        parsed.append(normalized in {"1", "true", "yes", "on"})
    return parsed


@click.command()
@click.option("--checkpoint-path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--text", type=str, required=True)
@click.option("--prompt-text", type=str, default=None, multiple=True)
@click.option("--prompt-tokens", type=click.Path(path_type=Path, exists=True), default=None, multiple=True)
@click.option("--device", type=str, default="cuda")
@click.option("--half/--no-half", default=False)
@click.option("--max-new-tokens", type=int, default=256)
@click.option("--top-p", type=float, default=0.9)
@click.option("--top-k", type=int, default=30)
@click.option("--temperature", type=float, default=1.0)
@click.option("--chunk-length", type=int, default=300)
@click.option("--optim-levels", type=str, default="0,1,2,3")
@click.option("--cudagraphs", type=str, default="0,1")
@click.option("--int8-backend", type=str, default="auto")
def main(
    checkpoint_path: Path,
    text: str,
    prompt_text: tuple[str, ...],
    prompt_tokens: tuple[Path, ...],
    device: str,
    half: bool,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
    chunk_length: int,
    optim_levels: str,
    cudagraphs: str,
    int8_backend: str,
) -> None:
    precision = torch.float16 if half else torch.bfloat16
    prompt_tokens_list = [torch.from_numpy(np.load(path)) for path in prompt_tokens]

    rows = []
    for optim_level in _parse_csv_ints(optim_levels):
        for use_cudagraphs in _parse_csv_bools(cudagraphs):
            config = get_runtime_config(
                optim_level=optim_level,
                use_cudagraphs=use_cudagraphs,
                int8_backend=int8_backend,
            )
            apply_runtime_config(config)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            t0 = time.perf_counter()
            model, decode_one_token = init_model(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=False,
            )
            load_sec = time.perf_counter() - t0

            generated_tokens = 0
            t1 = time.perf_counter()
            for response in generate_long(
                model=model,
                device=device,
                decode_one_token=decode_one_token,
                text=text,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                chunk_length=chunk_length,
                prompt_text=list(prompt_text) if prompt_text else None,
                prompt_tokens=prompt_tokens_list or None,
            ):
                if response.action == "sample" and response.codes is not None:
                    generated_tokens += response.codes.size(1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_sec = time.perf_counter() - t1
            tokens_sec = generated_tokens / latency_sec if latency_sec > 0 else 0.0
            rows.append(
                {
                    "optim": config.label,
                    "cudagraphs": "on" if use_cudagraphs else "off",
                    "int8": config.int8_backend,
                    "tokens_sec": tokens_sec,
                    "latency_sec": latency_sec,
                    "load_sec": load_sec,
                    "vram_gb": (
                        torch.cuda.max_memory_reserved() / 1e9
                        if torch.cuda.is_available()
                        else None
                    ),
                    "ram_gb": system_ram_used_gb(),
                }
            )
            del model, decode_one_token

    print("optim,cudagraphs,int8,tokens_sec,latency_sec,load_sec,vram_gb,ram_gb")
    for row in rows:
        print(
            "{optim},{cudagraphs},{int8},{tokens_sec:.2f},{latency_sec:.2f},"
            "{load_sec:.2f},{vram_gb},{ram_gb}".format(
                **{
                    **row,
                    "vram_gb": (
                        f"{row['vram_gb']:.2f}" if row["vram_gb"] is not None else "n/a"
                    ),
                    "ram_gb": (
                        f"{row['ram_gb']:.2f}" if row["ram_gb"] is not None else "n/a"
                    ),
                }
            )
        )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
