"""Thin adapter around Meta's official SAM2 video predictor."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray


def _torch_device(device: str | int) -> str:
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        return f"cuda:{int(device)}"
    return str(device)


def _to_numpy(value: object) -> NDArray[np.generic]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class SAM2OfficialVideoTracker:
    """Propagate accepted masks one adjacent keyframe with SAM2 memory."""

    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        *,
        device: str | int = 0,
        predictor_instance: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self._device = _torch_device(device)
        self._uses_injected_predictor = predictor_instance is not None
        if predictor_instance is None:
            try:
                from sam2.sam2_video_predictor import SAM2VideoPredictor
            except ImportError as error:
                raise ImportError(
                    "SAM2 video tracklets require Meta's official 'sam2' package"
                ) from error
            predictor_instance = SAM2VideoPredictor.from_pretrained(
                model_id, device=self._device
            )
        self._predictor = predictor_instance

    @contextmanager
    def _inference_context(self) -> Iterator[None]:
        if self._uses_injected_predictor:
            yield
            return
        import torch

        with torch.inference_mode():
            if self._device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    yield
            else:
                yield

    def initialize(
        self,
        video_directory: str,
        *,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = False,
    ) -> object:
        with self._inference_context():
            return self._predictor.init_state(
                video_path=video_directory,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
            )

    def propagate_one_step(
        self,
        inference_state: object,
        *,
        source_frame_index: int,
        source_masks: Sequence[NDArray[np.bool_]],
        max_seeds_per_batch: int = 16,
    ) -> tuple[NDArray[np.bool_], ...]:
        """Return masks at ``source_frame_index + 1`` in source-mask order."""

        if max_seeds_per_batch <= 0:
            raise ValueError("max_seeds_per_batch must be positive")
        masks = tuple(np.asarray(mask, dtype=bool) for mask in source_masks)
        if not masks:
            return ()
        propagated: list[NDArray[np.bool_] | None] = [None] * len(masks)
        target_frame_index = source_frame_index + 1

        for batch_start in range(0, len(masks), max_seeds_per_batch):
            batch_stop = min(batch_start + max_seeds_per_batch, len(masks))
            self._predictor.reset_state(inference_state)
            with self._inference_context():
                for local_index, mask in enumerate(
                    masks[batch_start:batch_stop]
                ):
                    self._predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=source_frame_index,
                        obj_id=local_index,
                        mask=mask,
                    )
                outputs = self._predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=source_frame_index,
                    max_frame_num_to_track=1,
                    reverse=False,
                )
                target_output = None
                for frame_index, object_ids, mask_logits in outputs:
                    if int(frame_index) == target_frame_index:
                        target_output = (object_ids, mask_logits)
                        break
            if target_output is None:
                raise RuntimeError(
                    f"SAM2 did not return target keyframe index {target_frame_index}"
                )
            object_ids, mask_logits = target_output
            logits = _to_numpy(mask_logits)
            if logits.ndim == 4 and logits.shape[1] == 1:
                logits = logits[:, 0]
            if logits.ndim != 3:
                raise ValueError(
                    "SAM2 mask logits must have shape (N,H,W); "
                    f"got {logits.shape}"
                )
            for output_index, object_id in enumerate(object_ids):
                source_index = batch_start + int(object_id)
                if not batch_start <= source_index < batch_stop:
                    raise ValueError(f"SAM2 returned unknown object ID {object_id}")
                propagated[source_index] = np.ascontiguousarray(
                    logits[output_index] > 0.0
                )

        if any(mask is None for mask in propagated):
            raise RuntimeError("SAM2 omitted one or more propagated objects")
        return tuple(mask for mask in propagated if mask is not None)
