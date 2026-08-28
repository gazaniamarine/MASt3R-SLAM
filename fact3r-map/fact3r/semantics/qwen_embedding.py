"""Qwen3-VL-Embedding adapter for mask-level semantic retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def _normalise_rows(values: object) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("Qwen embeddings must be a finite matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Qwen embeddings cannot contain zero rows")
    return np.ascontiguousarray(array / norms)


class Qwen3VLEmbeddingEncoder:
    """Use the official embedding checkpoint without text generation.

    Qwen's retrieval model represents a multimodal input with the final
    non-padding hidden state.  Images and queries therefore occupy the same
    normalized space, unlike raw Qwen instruction-model vision tokens.
    """

    index_format = "fact3r-qwen-embedding-observation-index"
    semantic_query_capable = True

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        min_pixels: int = 128 * 128,
        max_pixels: int = 224 * 224,
        attention_implementation: str | None = None,
        query_instruction: str = (
            "Retrieve object images relevant to the robot user's query."
        ),
        image_instruction: str = "Represent the highlighted object image.",
    ) -> None:
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("Qwen embedding pixel limits are invalid")
        if dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("unsupported Qwen embedding dtype")
        started = perf_counter()
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Qwen3-VL-Embedding requires transformers>=4.57.3, torch, "
                "accelerate, and Pillow"
            ) from error

        model_kwargs: dict[str, object] = {
            "device_map": device_map,
            "dtype": dtype if dtype == "auto" else getattr(torch, dtype),
        }
        if attention_implementation is not None:
            model_kwargs["attn_implementation"] = attention_implementation
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            model_name,
            padding_side="right",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._torch = torch
        self._model_name = model_name
        self._device_name = str(self._model.device)
        self._query_instruction = query_instruction
        self._image_instruction = image_instruction
        self._load_seconds = perf_counter() - started

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    @staticmethod
    def _conversation(
        content: Sequence[Mapping[str, object]], instruction: str
    ) -> list[dict[str, object]]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction}],
            },
            {"role": "user", "content": list(content)},
        ]

    def _encode(self, conversations: Sequence[object]) -> NDArray[np.float32]:
        if not conversations:
            return np.empty((0, 0), dtype=np.float32)
        inputs = self._processor.apply_chat_template(
            list(conversations),
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self._model.device)
        with self._torch.inference_mode():
            outputs = self._model.model(**inputs, use_cache=False, return_dict=True)
        attention = inputs["attention_mask"]
        last_positions = attention.shape[1] - attention.flip(dims=[1]).argmax(dim=1) - 1
        rows = self._torch.arange(
            outputs.last_hidden_state.shape[0], device=outputs.last_hidden_state.device
        )
        embeddings = outputs.last_hidden_state[rows, last_positions]
        return _normalise_rows(embeddings.detach().float().cpu().numpy())

    def encode_images(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        conversations = [
            self._conversation(
                [{"type": "image", "image": image.convert("RGB")}],
                self._image_instruction,
            )
            for image in images
        ]
        return self._encode(conversations)

    def encode_text(self, texts: Sequence[str]) -> NDArray[np.float32]:
        conversations = [
            self._conversation(
                [{"type": "text", "text": text}], self._query_instruction
            )
            for text in texts
        ]
        return self._encode(conversations)
