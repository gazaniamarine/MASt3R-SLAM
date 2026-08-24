# Milestone 0 — frozen interface and geometric regression

## Coordinate and array conventions

- Image-aligned arrays use `(row, column)` indexing and `(height, width, ...)` shapes.
- `KeyframeRecord.pointmap_camera[row, column]` is an XYZ point in the keyframe/camera coordinate frame.
- `KeyframeRecord.pose_world_from_camera` is a homogeneous 4x4 transform. Its linear 3x3 block may contain Sim(3) scale.
- World points are computed as `x_world = linear @ x_camera + translation`.
- `mast3r_descriptors` means the dense downstream MASt3R descriptor map `D`, not the encoder token tensor stored as `Frame.feat`.
- Geometry confidence `C` and descriptor confidence `Q` remain separate fields.
- `LiftedProposal.pixel_rc[n]`, `points_world[n]`, RGB, confidence, and descriptor rows all refer to the same selected source pixel.

## Implemented interfaces

- `KeyframeRecord`: immutable, validated keyframe data consumed by Fact3R.
- `LiftedProposal`: immutable, point-aligned lifted-mask data.
- `Entity`: validated persistent object/part state container.
- `SemanticFact`: validated grounded fact state container.
- `keyframe_record_from_mast3r`: read-only adapter for an existing MASt3R-SLAM `Frame`.
- `lift_mask_to_3d`: confidence-filtered mask lifting into world coordinates.

The adapter accepts `D` and `Q` explicitly because the current parent `Frame` does not retain them. It never substitutes `Frame.feat` for `D`.

## Deterministic regression sequence

`tests/fixtures/milestone0_sequence.json` contains two 2x3 keyframes viewing the same planar patch. The second camera is translated by +0.1 m along world X, while its local pointmap is offset by -0.1 m. The selected mask must therefore lift to exactly the same four world points in both views.

Run the regression tests from `fact3r-map`:

```bash
python -m unittest discover -s tests -v
```

Generate a PLY in which the two pointmaps share world coordinates and the two masks are coloured magenta and cyan:

```bash
python scripts/visualize_milestone0.py
```

The default output is `artifacts/milestone0_alignment.ply`. Both lifted proposals must report the world centroid `(0.150, 0.050, 1.000)`.

## Exit condition

The numerical regression asserts that masks selected in two camera frames lift to the same expected world patch. The PLY exporter provides the corresponding visual check in the global reconstruction frame.
