"""Lazy BLIP wrapper for short region captions.

BLIP is the "small region captioner" tier: SigLIP stays the retrieval/
grouping workhorse (untouched), Qwen stays reserved for judgment calls
(verifying an ambiguous SigLIP match, picking a mark during live pixel-goal
selection), and BLIP fills the description tier -- "what is this crop, in
words" -- for both per-entity captions and per-region frame captions.

Florence-2 was tried first for this role and rejected: its hosted checkpoint
(microsoft/Florence-2-base) fails to load under current transformers on
either its legacy trust_remote_code path or the newer native
Florence2ForConditionalGeneration path (three distinct, unrelated attribute
errors were reproduced) -- the checkpoint's files have not kept pace with
transformers' internals. BLIP loads and captions correctly out of the box.
"""

from __future__ import annotations

from time import perf_counter
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


def _rgb_uint8(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[-1] < 3:
        raise ValueError("image must have shape (height, width, >=3)")
    values = values[:, :, :3]
    if np.issubdtype(values.dtype, np.floating) and values.size:
        if float(np.nanmax(values)) <= 1.0:
            values = values * 255.0
    return np.ascontiguousarray(np.clip(values, 0, 255).astype(np.uint8))


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BlipCaptioner:
    """Lazy Hugging Face adapter for BLIP image captioning.

    Mirrors fact3r.semantics.vlm_verification.Qwen3VLVerifier's shape (lazy
    load, model_name/load_seconds properties) for consistency, but the model
    itself is far smaller and has no chat/JSON-structured output -- it always
    returns a short free-text caption.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        *,
        device: str = "auto",
        max_new_tokens: int = 30,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self._model_name = model_name
        self._device = device
        self._max_new_tokens = max_new_tokens
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
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as error:
            raise RuntimeError(
                "BLIP captioning requires transformers, torch, and Pillow"
            ) from error
        self._resolved_device = _resolve_device(self._device)
        self._processor = BlipProcessor.from_pretrained(self._model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(
            self._model_name, dtype=torch.float32
        )
        self._model.to(self._resolved_device)
        self._model.eval()
        self._load_seconds = perf_counter() - started

    def caption(
        self, image: Union[NDArray[np.generic], "PIL.Image.Image"],  # noqa: F821
        *, prompt: Optional[str] = None,
    ) -> str:
        """One short caption for one crop. prompt conditions it (e.g. "a photo of")."""

        self._ensure_loaded()
        import torch
        from PIL import Image

        pil_image = (
            image if isinstance(image, Image.Image)
            else Image.fromarray(_rgb_uint8(image))
        )
        if prompt:
            inputs = self._processor(images=pil_image, text=prompt, return_tensors="pt")
        else:
            inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self._resolved_device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        caption = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return caption.strip()
