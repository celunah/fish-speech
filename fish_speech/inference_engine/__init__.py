import gc
import queue
import threading
from typing import Generator

import numpy as np
import torch
from loguru import logger

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.utils import InferenceResult, wav_chunk_header
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
)
from fish_speech.utils import autocast_exclude_mps, set_seed
from fish_speech.utils.schema import ServeTTSRequest


class TTSInferenceEngine(ReferenceLoader, VQManager):

    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: DAC,
        precision: torch.dtype,
        compile: bool,
    ) -> None:

        super().__init__()

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.llama_device = getattr(llama_queue, "device", decoder_model.device)
        self.precision = precision
        self.compile = compile
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            self.llama_queue.put(None)
            worker_thread = getattr(self.llama_queue, "worker_thread", None)
            if worker_thread is not None:
                worker_thread.join(timeout=30)
        finally:
            self.ref_by_id.clear()
            self.ref_by_hash.clear()
            self.decoder_model = None
            try:
                from tools.llama.quantize import clear_int8_dequant_cache

                clear_int8_dequant_cache()
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            gc.collect()

    @torch.inference_mode()
    def inference(self, req: ServeTTSRequest) -> Generator[InferenceResult, None, None]:
        """
        Main inference function:
        - Loads the reference audio and text.
        - Calls the LLAMA model for inference.
        - Decodes the VQ tokens to audio.
        """
        if self._closed:
            raise RuntimeError("TTSInferenceEngine is closed")

        ref_id: str | None = req.reference_id
        prompt_tokens, prompt_texts = [], []
        # Load the reference audio and text based on id or hash
        if ref_id is not None:
            prompt_tokens, prompt_texts = self.load_by_id(ref_id, req.use_memory_cache)

        elif req.references:
            prompt_tokens, prompt_texts = self.load_by_hash(
                req.references, req.use_memory_cache
            )

        # Set the random seed if provided
        if req.seed is not None:
            set_seed(req.seed)
            logger.warning(f"set seed: {req.seed}")

        # Get the symbolic tokens from the LLAMA model
        response_queue, cancel_event = self.send_Llama_request(
            req, prompt_tokens, prompt_texts
        )

        # Get the sample rate from the decoder model
        if hasattr(self.decoder_model, "spec_transform"):
            sample_rate = self.decoder_model.spec_transform.sample_rate
        else:
            sample_rate = self.decoder_model.sample_rate

        # If streaming, send the header
        if req.streaming:
            yield InferenceResult(
                code="header",
                audio=(
                    sample_rate,
                    np.array(wav_chunk_header(sample_rate=sample_rate)),
                ),
                error=None,
            )

        segments = []

        try:
            while True:
                # Get the response from the LLAMA model
                wrapped_result: WrappedGenerateResponse = response_queue.get()
                if wrapped_result.status == "error":
                    yield InferenceResult(
                        code="error",
                        audio=None,
                        error=(
                            wrapped_result.response
                            if isinstance(wrapped_result.response, BaseException)
                            else Exception("Unknown error")
                        ),
                    )
                    break

                # Accept alias-loaded GenerateResponse objects too. When users import
                # through faster_fish_speech, Python can load the same source file
                # under both package names, making a strict isinstance check fail.
                if not isinstance(wrapped_result.response, GenerateResponse) and not (
                    type(wrapped_result.response).__name__ == "GenerateResponse"
                    and hasattr(wrapped_result.response, "action")
                    and hasattr(wrapped_result.response, "codes")
                    and hasattr(wrapped_result.response, "text")
                ):
                    raise TypeError(
                        f"Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                    )

                result = wrapped_result.response
                if result.action != "next":
                    segment = self.get_audio_segment(result)

                    if req.streaming:  # Used only by the API server
                        yield InferenceResult(
                            code="segment",
                            audio=(sample_rate, segment),
                            error=None,
                        )
                    segments.append(segment)
                else:
                    break
        finally:
            cancel_event.set()

        # Clean up the memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Edge case: no audio generated
        if len(segments) == 0:
            yield InferenceResult(
                code="error",
                audio=None,
                error=RuntimeError("No audio generated, please check the input text."),
            )
        else:
            # Streaming or not, return the final audio
            audio = np.concatenate(segments, axis=0)
            yield InferenceResult(
                code="final",
                audio=(sample_rate, audio),
                error=None,
            )

        return None

    def send_Llama_request(
        self, req: ServeTTSRequest, prompt_tokens: list, prompt_texts: list
    ) -> tuple[queue.Queue, threading.Event]:
        """
        Send a request to the LLAMA model to generate the symbolic tokens.
        """

        # Prepare the request
        request = dict(
            device=self.llama_device,
            max_new_tokens=req.max_new_tokens,
            text=req.text,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            iterative_prompt=req.chunk_length > 0,
            chunk_length=req.chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
        )

        # Create a queue to get the response
        response_queue = queue.Queue()
        cancel_event = threading.Event()

        # Send the request to the LLAMA model
        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
                cancel_event=cancel_event,
            )
        )

        return response_queue, cancel_event

    def get_audio_segment(self, result: GenerateResponse) -> np.ndarray:
        """
        Decode the VQ tokens to audio.
        """

        # Don't use autocast on MPS devices
        with autocast_exclude_mps(
            device_type=self.decoder_model.device.type, dtype=self.precision
        ):
            # Decode the symbolic tokens to audio
            segment = self.decode_vq_tokens(codes=result.codes)

        # Convert the audio to numpy
        return segment.float().cpu().numpy()
