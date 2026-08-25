# Fact3R-Map implementation status

Snapshot: 2026-08-25
Audited MASt3R-SLAM commit: `e6f4e3d` (`Update bibtex`)

This document distinguishes reusable functionality in the parent MASt3R-SLAM repository from functionality implemented specifically for Fact3R-Map.

## Summary

The parent repository implements the MASt3R-SLAM geometry and tracking backbone. The new `fact3r-map` package completes Milestone 0 and includes Hungarian, balanced-transport, and visibility-conditioned unbalanced proposal-to-entity association stages. It does **not** yet implement delayed persistent entity lifecycle updates or semantic mapping; Milestones 1–5 remain open.

The reusable boundary is:

```text
RGB input -> MASt3R inference -> keyframes -> pointmaps/confidence
          -> tracking/global poses -> loop closure -> reconstruction
```

Fact3R now has validated keyframe, lifted-proposal, entity, and semantic-fact contracts; a read-only keyframe adapter; a finalized-keyframe disk export; official and Transformers SAM2 automatic mask generation; geometry-aware mask filtering; mask-to-world lifting; alignment and RGB association visualization; saved-proposal streaming; re-anchored official-SAM2 video tracklets; spatial proposal-entity gating; reusable multi-cue pairwise costs with an optional IoU-weighted temporal cue; exact Hungarian assignment; balanced log-domain Sinkhorn transport; projected entity visibility; and strict-support visibility-conditioned unbalanced transport with directional birth/miss residuals. Persistent confirmation/inactive lifecycle behavior, delayed commitment, confidence-gated memory, semantic extraction, BEV planning, and semantic querying still need to be implemented.

## Reusable MASt3R-SLAM foundations

| Capability required by Fact3R-Map | Status | Existing implementation and limitation |
|---|---|---|
| Monocular RGB video/image input | Implemented | `mast3r_slam/dataloader.py` and `main.py` accept videos, image folders, datasets, and a RealSense stream. |
| Keyframe creation and selection | Implemented | `mast3r_slam/frame.py`, `mast3r_slam/tracker.py`, and `main.py` maintain shared keyframes. |
| Camera poses | Implemented | Each keyframe stores `T_WC`; tracking and global optimization update Sim(3) poses. |
| Dense pointmaps | Implemented | Each frame/keyframe stores `X_canon`; world points can be obtained with `T_WC.act(X_canon)`. |
| Geometry confidence | Implemented | Each frame/keyframe stores and fuses `C`; `get_average_conf()` exposes the current confidence. |
| RGB aligned with pointmap pixels | Implemented in memory | `uimg` is stored at pointmap resolution. RGB keyframe images can also be exported. |
| MASt3R dense descriptors and descriptor confidence | Partially implemented | Inference computes `D` and `Q` in `mast3r_slam/mast3r_utils.py`, and matching consumes them. `D` is not retained in `Frame`/`SharedKeyframes` or exported. `Q` is used transiently and in factor-graph correspondence weights, but a per-keyframe descriptor-confidence map is not retained. The stored encoder token `feat` is not the same as the dense downstream descriptor `D`. |
| Dense MASt3R correspondence machinery | Partially reusable | Directional dense matches, validity masks, and confidence checks exist in `mast3r_slam/matching.py` and `mast3r_slam/global_opt.py`. There is no proposal-to-entity reciprocal match ratio or entity descriptor bank. |
| Loop closure and relocalization | Implemented | Retrieval-driven loop closure/relocalization and global pose optimization exist in `main.py` and `mast3r_slam/global_opt.py`. |
| Global coloured reconstruction | Implemented | `mast3r_slam/evaluate.py` transforms keyframe pointmaps to world space and exports a confidence-filtered coloured PLY. |
| 3D visualization | Partially reusable | `mast3r_slam/visualization.py` renders camera poses and pointmaps/surfels. It has no mask, proposal, entity-ID, or semantic-fact overlays. |
| Regression/evaluation infrastructure | Partially reusable | Dataset loaders and SLAM trajectory evaluation scripts exist, but there are no Fact3R entity, semantic, or navigation metrics/tests. |

## Fact3R-Map milestone audit

| Milestone | Current status | Missing work |
|---|---|---|
| Milestone 0 — Freeze the interface | Complete | Added validated `KeyframeRecord`, `LiftedProposal`, `Entity`, and `SemanticFact` contracts; a read-only MASt3R frame adapter with explicit `D/Q` inputs; confidence-filtered mask lifting; a deterministic two-view sequence; numerical world-alignment tests; and a coloured PLY alignment exporter. The synthetic exit condition passes. Real-sequence model validation will be performed when Milestone 1 supplies masks. |
| Milestone 1 — Entity mapping without learning | In progress | Implemented official/Transformers SAM2 proposal generation, filtering, offline keyframe export, 3D lifting and storage, saved-frame loading, adjacent-keyframe SAM2 mask propagation, one-to-one proposal tracklets, spatial gating, normalized geometry/colour/descriptor/temporal costs, exact Hungarian assignment, immediate provisional-entity creation, geometry update, frame/entity manifests, unmatched-reason diagnostics, and persistent-ID RGB/GIF visualization. Still missing confirmation/inactive lifecycle rules, real-sequence tracklet/threshold calibration, and complete-frame MASt3R vote aggregation. |
| Milestone 2 — Split/merge robustness | Not started | No mixed-mask splitting, fragment merging, part hierarchy, inactive state, or entity reactivation. |
| Milestone 3 — Unbalanced transport | In progress | Implemented balanced log-domain Sinkhorn and a separate generalized Sinkhorn solver with per-proposal and per-entity KL relaxation. Entity demand is conditioned on current-view projection/depth visibility; proposal evidence uses lifted-point retention and optional tracklet confidence; forbidden edges retain exactly zero mass; directional birth/miss residuals, excess mass, confidence and rejection reasons are saved. A shared dustbin is retained only as a future ablation. Still missing a real-scene run, delayed commitment, confidence-gated memory, and full Hungarian-vs-UOT evaluation. |
| Milestone 4 — Semantic facts | Not started | No view selection, structured fact extraction/fusion, point/part grounding, fact graph, or inspection tooling. |
| Milestone 5 — Long-horizon retrieval and return | Not started | No BEV occupancy/traversability map, topology, entity-to-BEV links, goal-pose generation, query parser, graph matching, or navigation evaluation. |

## Components that do not currently exist

The parent repository contains no non-third-party implementation matching these Fact3R concepts:

- persistent object or part entities;
- voxel filtering and stable mask-hierarchy extraction;
- delayed association belief and confidence-gated entity lifecycle updates;
- temporal association beliefs, split/merge handling, or lifecycle states;
- semantic fact extraction, fusion, provenance, or graph storage;
- open-vocabulary semantic query parsing/retrieval;
- BEV occupancy, traversability, topology, or semantic goal planning.

## First implementation step to take next

Run `scripts/run_visibility_residual_transport.py` over the existing HM3D keyframes, proposals and tracklets. Compare entity growth, birth/miss residuals, ambiguity, temporal-hint agreement and visibility against the 399-entity geometry-only, 344-entity tracklet-Hungarian, and 338-entity balanced-Sinkhorn runs. Then use the saved residuals for delayed commitment; a shared learned dustbin should be evaluated only as a controlled ablation.
