"""Lazy SmolVLM-Instruct wrapper, a drop-in alternative to BlipCaptioner.

Same "small region captioner" tier as blip_captioner.py (see that module's
docstring for the SigLIP/Qwen/BLIP division of labour this slots into) --
but unlike BLIP's plain conditional-generation captioning, SmolVLM-Instruct
is an instruction-tuned chat model, so `caption` builds a one-turn chat
prompt via apply_chat_template, the same pattern nav_pipeline/qwen_pixel_
goal.py already uses for Qwen2.5-VL/Qwen3-VL, rather than calling generate
directly on the image.

Two environment facts baked in here, both confirmed on this machine, not
assumed:
  - `AutoModelForVision2Seq` (the class HuggingFace's own SmolVLM examples
    use) does not exist in this environment's transformers (5.12.1) --
    it was renamed to `AutoModelForImageTextToText`.
  - `flash_attn` is not installed here, so `attn_implementation` always
    resolves to "eager" regardless of device; a from_pretrained call with
    "flash_attention_2" would fail outright, not silently fall back.
"""

from __future__ import annotations

from time import perf_counter
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from .blip_captioner import _resolve_device, _rgb_uint8


class SmolVLMCaptioner:
    """Lazy Hugging Face adapter for SmolVLM-Instruct captioning.

    Same lazy-load shape as BlipCaptioner (model_name/load_seconds
    properties, `caption(image, prompt=None) -> str`) so it is a drop-in
    swap wherever a captioner is used.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM-Instruct",
        *,
        device: str = "auto",
        max_new_tokens: int = 40,
        prompt: str = "Describe this image in one short sentence.",
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self._model_name = model_name
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._default_prompt = prompt
        self._model = None
        self._processor = None
        self._resolved_device = None
        self._load_seconds = 0.0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    def load(self) -> None:
        """Load the model now so interactive/batch use never pays cold start mid-run."""

        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        started = perf_counter()
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "SmolVLM captioning requires transformers and torch"
            ) from error
        self._resolved_device = _resolve_device(self._device)
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        dtype = torch.bfloat16 if self._resolved_device == "cuda" else torch.float32
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_name, dtype=dtype, attn_implementation="eager",
        ).to(self._resolved_device)
        self._model.eval()
        self._load_seconds = perf_counter() - started

    def caption(
        self, image: Union[NDArray[np.generic], "PIL.Image.Image"],  # noqa: F821
        *, prompt: Optional[str] = None,
    ) -> str:
        """One short caption for one crop. prompt overrides the default instruction."""

        self._ensure_loaded()
        import torch
        from PIL import Image

        pil_image = (
            image if isinstance(image, Image.Image)
            else Image.fromarray(_rgb_uint8(image))
        )
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt or self._default_prompt},
        ]}]
        chat_text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=chat_text, images=[pil_image], return_tensors="pt"
        ).to(self._resolved_device)
        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        caption = self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return caption.strip()
