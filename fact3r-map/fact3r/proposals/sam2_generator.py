"""Optional Hugging Face SAM2 automatic-mask-generation backend."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fact3r.proposals.mask_generator import MaskProposal2D


def _to_numpy(value: Any) -> NDArray[np.generic]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _rgb_uint8(rgb: NDArray[np.generic]) -> NDArray[np.uint8]:
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("SAM2 input RGB must have shape (height, width, 3)")
    if np.issubdtype(values.dtype, np.floating) and values.size:
        if float(np.nanmax(values)) <= 1.0:
            values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def _resize_binary_mask(mask: NDArray[np.bool_], shape: tuple[int, int]) -> NDArray[np.bool_]:
    if mask.shape == shape:
        return mask
    source_h, source_w = mask.shape
    target_h, target_w = shape
    rows = np.minimum(
        (np.arange(target_h) * source_h / target_h).astype(np.int64), source_h - 1
    )
    columns = np.minimum(
        (np.arange(target_w) * source_w / target_w).astype(np.int64), source_w - 1
    )
    return mask[rows[:, None], columns[None, :]]


def _pipeline_device(device: str | int) -> str | int:
    if isinstance(device, str) and device.isdigit():
        return int(device)
    return device


class SAM2AutomaticMaskGenerator:
    """Generate proposals over the whole image using SAM2's prompt grid.

    Unlike direct ``Sam2Model`` calls with point prompts, the Hugging Face
    ``mask-generation`` pipeline creates the grid prompts internally and returns
    a class-agnostic, overlapping set of candidate masks.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-large",
        *,
        device: str | int = 0,
        points_per_batch: int = 64,
        pred_iou_threshold: float = 0.88,
        stability_score_threshold: float = 0.95,
        pipeline_instance: Any | None = None,
    ) -> None:
        if points_per_batch < 1:
            raise ValueError("points_per_batch must be positive")
        self.model_id = model_id
        self.points_per_batch = points_per_batch
        self.pred_iou_threshold = pred_iou_threshold
        self.stability_score_threshold = stability_score_threshold
        if pipeline_instance is None:
            try:
                from transformers import pipeline
            except ImportError as error:
                raise ImportError(
                    "SAM2 support requires the 'sam2' optional dependencies"
                ) from error
            pipeline_instance = pipeline(
                task="mask-generation",
                model=model_id,
                device=_pipeline_device(device),
            )
        self._pipeline = pipeline_instance

    def generate(
        self, rgb: NDArray[np.generic], *, frame_id: int
    ) -> list[MaskProposal2D]:
        from PIL import Image

        rgb_uint8 = _rgb_uint8(rgb)
        image = Image.fromarray(rgb_uint8)
        outputs = self._pipeline(
            image,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_threshold,
            stability_score_thresh=self.stability_score_threshold,
            output_bboxes_mask=True,
        )
        masks = outputs.get("masks", [])
        scores = _to_numpy(outputs.get("scores", np.ones(len(masks)))).reshape(-1)
        boxes_value = outputs.get("bounding_boxes")
        boxes = None if boxes_value is None else _to_numpy(boxes_value)
        if len(scores) != len(masks):
            raise ValueError("SAM2 returned different mask and score counts")
        if boxes is not None and len(boxes) != len(masks):
            raise ValueError("SAM2 returned different mask and bounding-box counts")

        proposals = []
        for index, mask_value in enumerate(masks):
            mask = _to_numpy(mask_value).astype(bool, copy=False)
            while mask.ndim > 2 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 2:
                raise ValueError(f"SAM2 mask must be 2D; got {mask.shape}")
            mask = _resize_binary_mask(mask, rgb_uint8.shape[:2])
            proposals.append(
                MaskProposal2D(
                    proposal_id=f"frame-{frame_id:06d}-sam2-{index:04d}",
                    frame_id=frame_id,
                    mask=mask,
                    score=float(scores[index]),
                    bounding_box_xyxy=None if boxes is None else boxes[index],
                    source=f"sam2:{self.model_id}",
                    metadata={"raw_index": index},
                )
            )
        return proposals
