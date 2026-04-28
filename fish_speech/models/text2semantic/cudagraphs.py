from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger


@dataclass
class _GraphState:
    graph: torch.cuda.CUDAGraph
    static_inputs: tuple[torch.Tensor, ...]
    static_outputs: tuple[torch.Tensor, ...]
    result_type: type | None = None


class CudaGraphHotPath:
    """CUDA graph replay for fixed-shape token generation forwards.

    Capture is intentionally narrow: only the model forward kernels used after
    prompt prefill are graphed. Sampling, stop checks, reference handling,
    codecs, loading, and all CPU work stay in eager mode.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.enabled = True
        self._forward_generate: _GraphState | None = None
        self._forward_generate_fast: _GraphState | None = None
        self.slow_replays = 0
        self.fast_replays = 0
        self.eager_fallbacks = 0

    @staticmethod
    def can_enable(device: torch.device | str) -> bool:
        device = torch.device(device)
        return device.type == "cuda" and torch.cuda.is_available()

    def _disable(self, exc: BaseException) -> None:
        self.enabled = False
        logger.warning(f"CUDA graph capture disabled, falling back to eager mode: {exc}")

    def forward_generate(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        *,
        audio_masks: torch.Tensor | None = None,
        audio_parts: torch.Tensor | None = None,
    ):
        if (
            not self.enabled
            or not x.is_cuda
            or not input_pos.is_cuda
            or audio_masks is not None
            or audio_parts is not None
        ):
            self.eager_fallbacks += 1
            return self.model.forward_generate(
                x, input_pos, audio_masks=audio_masks, audio_parts=audio_parts
            )

        if self._forward_generate is None:
            try:
                self._forward_generate = self._capture_forward_generate(x, input_pos)
            except Exception as exc:
                self._disable(exc)
                return self.model.forward_generate(x, input_pos)

        state = self._forward_generate
        static_x, static_pos = state.static_inputs
        if static_x.shape != x.shape or static_pos.shape != input_pos.shape:
            self.eager_fallbacks += 1
            return self.model.forward_generate(x, input_pos)

        static_x.copy_(x)
        static_pos.copy_(input_pos)
        state.graph.replay()
        self.slow_replays += 1
        logits, hidden_states = state.static_outputs
        return state.result_type(logits=logits, hidden_states=hidden_states)

    def forward_generate_fast(
        self,
        hidden_states: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled or not hidden_states.is_cuda or not input_pos.is_cuda:
            self.eager_fallbacks += 1
            return self.model.forward_generate_fast(hidden_states, input_pos)

        if self._forward_generate_fast is None:
            try:
                self._forward_generate_fast = self._capture_forward_generate_fast(
                    hidden_states, input_pos
                )
            except Exception as exc:
                self._disable(exc)
                return self.model.forward_generate_fast(hidden_states, input_pos)

        state = self._forward_generate_fast
        static_hidden, static_pos = state.static_inputs
        if (
            static_hidden.shape != hidden_states.shape
            or static_pos.shape != input_pos.shape
        ):
            self.eager_fallbacks += 1
            return self.model.forward_generate_fast(hidden_states, input_pos)

        static_hidden.copy_(hidden_states)
        static_pos.copy_(input_pos)
        state.graph.replay()
        self.fast_replays += 1
        return state.static_outputs[0]

    def stats(self) -> str:
        if not self.enabled:
            return "disabled"
        return (
            f"slow_replays={self.slow_replays}, "
            f"fast_replays={self.fast_replays}, "
            f"fallbacks={self.eager_fallbacks}"
        )

    def _capture_forward_generate(
        self, x: torch.Tensor, input_pos: torch.Tensor
    ) -> _GraphState:
        static_x = x.detach().clone()
        static_pos = input_pos.detach().clone()

        result = self.model.forward_generate(static_x, static_pos)
        static_logits = torch.empty_like(result.logits)
        static_hidden = torch.empty_like(result.hidden_states)

        torch.cuda.synchronize(static_x.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = self.model.forward_generate(static_x, static_pos)
            static_logits.copy_(result.logits)
            static_hidden.copy_(result.hidden_states)

        logger.info("CUDA graph captured for slow token forward")
        return _GraphState(
            graph=graph,
            static_inputs=(static_x, static_pos),
            static_outputs=(static_logits, static_hidden),
            result_type=type(result),
        )

    def _capture_forward_generate_fast(
        self, hidden_states: torch.Tensor, input_pos: torch.Tensor
    ) -> _GraphState:
        static_hidden = hidden_states.detach().clone()
        static_pos = input_pos.detach().clone()

        result = self.model.forward_generate_fast(static_hidden, static_pos)
        static_logits = torch.empty_like(result)

        torch.cuda.synchronize(static_hidden.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = self.model.forward_generate_fast(static_hidden, static_pos)
            static_logits.copy_(result)

        logger.info("CUDA graph captured for fast codebook forward")
        return _GraphState(
            graph=graph,
            static_inputs=(static_hidden, static_pos),
            static_outputs=(static_logits,),
        )
