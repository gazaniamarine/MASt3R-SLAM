"""Automatic mask generation through Meta's official ``sam2`` package.

The Hugging Face ``mask-generation`` pipeline in :mod:`fact3r.proposals.sam2_generator`
and this backend solve the same problem. This one loads the checkpoint through
``sam2.automatic_mask_generator``, which is the reference implementation and the
path the ``facebook/sam2-hiera-large`` model card describes.

``SAM2ImagePredictor`` is deliberately not used here. It answers "what is at this
prompt", so it needs the caller to already know which objects matter. Mapping
needs every candidate object before any query exists, which is what the grid of
prompts driven by the automatic generator produces.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fact3r.proposals.mask_generator import MaskProposal2D


def _rgb_uint8(rgb: NDArray[np.generic]) -> NDArray[np.uint8]:
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("SAM2 input RGB must have shape (height, width, 3)")
    if np.issubdtype(values.dtype, np.floating) and values.size:
        if float(np.nanmax(values)) <= 1.0:
            values = values * 255.0
    return np.ascontiguousarray(np.clip(values, 0, 255).astype(np.uint8))


class SAM2OfficialMaskGenerator:
    """Class-agnostic proposals from ``sam2.automatic_mask_generator``.

    The generator is run under ``torch.inference_mode`` and bfloat16 autocast on
    CUDA, matching the reference usage. Masks come back at the resolution of the
    image that was passed in, so callers must pass the keyframe RGB rather than
    the original capture: MASt3R crops its input, and only the keyframe raster
    shares a pixel grid with the pointmap that the masks are lifted through.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        *,
        device: str | int = 0,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_threshold: float = 0.88,
        stability_score_threshold: float = 0.95,
        min_mask_region_area: int = 0,
        generator_instance: Any | None = None,
    ) -> None:
        if points_per_batch < 1:
            raise ValueError("points_per_batch must be positive")
        self.model_id = model_id
        self.points_per_batch = points_per_batch
        self.pred_iou_threshold = pred_iou_threshold
        self.stability_score_threshold = stability_score_threshold
        self._device = _torch_device(device)
        self._uses_injected_generator = generator_instance is not None
        if generator_instance is None:
            try:
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            except ImportError as error:
                raise ImportError(
                    "the official SAM2 backend requires the 'sam2' package "
                    "(pip install git+https://github.com/facebookresearch/sam2.git)"
                ) from error
            generator_instance = SAM2AutomaticMaskGenerator.from_pretrained(
                model_id,
                device=self._device,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_threshold,
                stability_score_thresh=stability_score_threshold,
                min_mask_region_area=min_mask_region_area,
            )
        self._generator = generator_instance

    def generate(
        self, rgb: NDArray[np.generic], *, frame_id: int
    ) -> list[MaskProposal2D]:
        image = _rgb_uint8(rgb)
        if self._uses_injected_generator:
            annotations = self._generator.generate(image)
        else:
            import torch

            use_autocast = str(self._device).startswith("cuda")
            with torch.inference_mode():
                if use_autocast:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        annotations = self._generator.generate(image)
                else:
                    annotations = self._generator.generate(image)

        proposals: list[MaskProposal2D] = []
        for index, annotation in enumerate(annotations):
            mask = np.asarray(annotation["segmentation"]).astype(bool, copy=False)
            if mask.shape != image.shape[:2]:
                raise ValueError(
                    f"SAM2 mask shape {mask.shape} does not match the "
                    f"input image {image.shape[:2]}"
                )
            box = annotation.get("bbox")
            proposals.append(
                MaskProposal2D(
                    proposal_id=f"frame-{frame_id:06d}-sam2-{index:04d}",
                    frame_id=frame_id,
                    mask=mask,
                    score=float(annotation["predicted_iou"]),
                    bounding_box_xyxy=None if box is None else _xywh_to_xyxy(box),
                    source=f"sam2-official:{self.model_id}",
                    metadata={
                        "raw_index": index,
                        "stability_score": float(annotation["stability_score"]),
                        "sam2_area": int(annotation["area"]),
                    },
                )
            )
        return proposals


def _xywh_to_xyxy(box: object) -> NDArray[np.float32]:
    x, y, width, height = np.asarray(box, dtype=np.float32).reshape(4)
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def _torch_device(device: str | int) -> str:
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        return f"cuda:{int(device)}"
    return str(device)
