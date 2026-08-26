# Fact3R-Map

**MASt3R-grounded, open-vocabulary semantic mapping with persistent 3D entities and unbalanced optimal-transport association**

## 1. Project summary

Fact3R-Map is a semantic mapping system for long-horizon vision-and-language navigation from monocular RGB video. It is intended for cases where an agent may travel a long distance and later receive an arbitrary semantic query such as:

- “Return to the black chair.”
- “Go back to the pink door with a metal handle.”
- “Find the damaged wooden cabinet beside the exit sign.”

The system does **not** store a dense CLIP embedding at every map point or BEV cell. Instead, it builds:

1. a metric occupancy/traversability map for navigation;
2. a persistent object-and-part entity map grounded in MASt3R reconstruction;
3. an open-vocabulary semantic fact graph attached to those entities.

The main research component is persistent 3D entity association under fragmented masks, partial visibility, repeated objects and long temporal gaps. The proposed association method uses confidence-aware **unbalanced optimal transport (UOT)** over geometric and MASt3R correspondence evidence.

## 2. Core research question

> Can dense MASt3R correspondences and 3D reconstruction be used to construct stable object identities over long trajectories, allowing detailed open-vocabulary semantic facts to be stored once per physical entity rather than as noisy dense visual-language features?

The central hypothesis is that MASt3R should answer:

> “Are these observations of the same physical thing, and where is it?”

A multi-view semantic model should answer:

> “What facts can we reliably state about that thing?”

Separating identity from semantics should make the map more stable, compact and interpretable than directly averaging CLIP features into a BEV.

## 3. Intended contribution

The proposed contribution is:

> A MASt3R-supervised, unbalanced optimal-transport method for persistent 3D entity association, followed by multi-view probabilistic consolidation of point-grounded, open-vocabulary semantic propositions.

The work should **not** claim the following as novel by themselves:

- MASt3R reconstruction or SLAM;
- lifting 2D masks into 3D;
- using a scene graph for semantic navigation;
- attaching captions or visual-language features to 3D objects;
- using BEV occupancy for path planning.

Related systems already cover parts of this pipeline, including [MASt3R-SLAM](https://openaccess.thecvf.com/content/CVPR2025/papers/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.pdf), [ConceptGraphs](https://arxiv.org/abs/2309.16650), [HOV-SG](https://arxiv.org/abs/2403.17846), [OVI-MAP](https://arxiv.org/abs/2603.26541), [SAB3R](https://arxiv.org/abs/2506.02112) and [LangSplat](https://openaccess.thecvf.com/content/CVPR2024/papers/Qin_LangSplat_3D_Language_Gaussian_Splatting_CVPR_2024_paper.pdf).

## 4. Scope

### Initial scope

- Monocular RGB video.
- Static indoor scenes.
- Globally aligned MASt3R pointmaps and poses.
- Object-level entities, plus a limited object-part hierarchy.
- Arbitrary semantic queries issued after the map has been built.
- No persistent storage of original RGB frames after semantic consolidation.
- Standard BEV path planning after semantic target retrieval.

### Initial non-goals

- Fully dynamic-world mapping.
- Manipulation or articulation modelling.
- End-to-end navigation-policy learning.
- Perfect recognition of every visible detail.
- Replacing MASt3R-SLAM.
- Building a photorealistic neural renderer of the complete environment.

## 5. System overview

```text
Monocular RGB video
        │
        ▼
MASt3R-SLAM
  pointmaps, poses, descriptors, confidence
        │
        ├──────────────► BEV occupancy and topology
        │
        ▼
Class-agnostic 2D masks
        │
        ▼
Lift masks into globally aligned 3D proposals
        │
        ▼
Geometric candidate gating
        │
        ▼
Point-level correspondence scoring
        │
        ▼
Unbalanced Sinkhorn mask-to-entity association
        │
        ▼
Temporal belief accumulation and hard commitment
        │
        ▼
Persistent 3D entity and part map
        │
        ▼
Multi-view structured semantic extraction
        │
        ▼
Point-grounded semantic fact graph
        │
        ▼
Text query → graph retrieval → entity location → BEV planning
```

## 6. Persistent map representation

### 6.1 Navigation layer

The BEV is used for geometry and planning, not as the primary semantic memory.

Each occupied or observed BEV cell stores:

```text
BEVCell:
    occupancy_probability
    minimum_height
    maximum_height
    traversability_cost
    room_or_region_id
    nearby_entity_ids[]
```

### 6.2 Entity layer

```text
Entity:
    id
    status                  # provisional, confirmed, inactive, dynamic
    parent_region_id
    parent_entity_id        # optional object-part hierarchy

    centroid_xyz
    bounding_box_xyz
    surfel_or_voxel_geometry
    normal_statistics
    colour_statistics

    mast3r_descriptor_bank
    descriptor_confidence
    observation_count
    observed_view_directions
    best_observation_pose

    first_seen_timestamp
    last_seen_timestamp
    persistence_probability

    semantic_fact_ids[]
```

### 6.3 Semantic fact layer

```text
SemanticFact:
    id
    entity_id
    subject
    predicate
    value

    support_type            # entire entity, part ID, sparse points, local 3D box
    support_reference

    posterior_probability
    supporting_view_count
    contradictory_view_count
    first_seen_timestamp
    last_seen_timestamp
    provenance_summary
```

Example:

```text
entity_42:
    type = chair                         confidence 0.96
    colour = black                       confidence 0.89
    material(seat) = fabric              confidence 0.84
    material(legs) = metal               confidence 0.87
    has_part = armrests                  confidence 0.78
    relation = beside(pink_door_7)       geometry-derived
```

## 7. Building the 3D entity map

### 7.1 Keyframe outputs

For each keyframe \(I_t\), retain during mapping:

- world-aligned pointmap \(X_t(u)\);
- geometry confidence \(C_t(u)\);
- dense MASt3R descriptor \(D_t(u)\);
- descriptor confidence \(Q_t(u)\);
- camera pose \(T_t\);
- RGB values associated with pointmap pixels.

For pixel \(u\), the world point is:

\[
x_u^W = T_t X_t(u)
\]

### 7.2 Class-agnostic proposals

Generate masks at one or more scales without assigning semantic classes. Before lifting a mask:

1. erode uncertain boundaries;
2. remove low-confidence MASt3R points;
3. voxelize the lifted points;
4. remove very small and spatially scattered components;
5. retain mask hierarchy information when a stable part/object relation exists.

For mask \(m_t^k\):

\[
P_t^k =
\left\{
(x_u^W,D_t(u),RGB_t(u))
\mid
u\in m_t^k,
C_t(u)>\tau_C,
Q_t(u)>\tau_Q
\right\}
\]

Each lifted mask begins as a provisional 3D entity proposal.

### 7.3 Candidate gating

Do not construct a dense cost matrix against every map entity. For a proposal \(m_i\), retain candidate entities satisfying at least one of:

- expanded 3D bounding boxes overlap;
- proposal centroid lies within a configurable spatial radius;
- reciprocal MASt3R matches connect the proposal to the entity;
- the entity projects into the proposal's current viewing region;
- a loop-closure keyframe connects their observation histories.

All impossible pairs receive infinite cost before assignment.

## 8. Association formulation

### 8.1 Pairwise affinity

For proposal \(m_i\) and persistent entity \(E_j\), construct:

\[
C_{ij} =
\lambda_1(1-\rho^{3R}_{ij}) +
\lambda_2(1-\rho^{xyz}_{ij}) +
\lambda_3(1-\mathrm{IoU}^{proj}_{ij}) +
\lambda_4\Delta^{shape}_{ij} +
\lambda_5\Delta^{colour}_{ij} +
\lambda_6\Delta^{time}_{ij}
\]

Where:

- \(\rho^{3R}_{ij}\): reciprocal MASt3R descriptor-match ratio;
- \(\rho^{xyz}_{ij}\): geometrically consistent point fraction;
- \(\mathrm{IoU}^{proj}_{ij}\): current mask versus projected entity overlap;
- \(\Delta^{shape}_{ij}\): disagreement in extent, normals and shape;
- \(\Delta^{colour}_{ij}\): disagreement in robust colour statistics;
- \(\Delta^{time}_{ij}\): temporal or motion inconsistency.

Colour and shape are supporting cues. They must not override strong geometric contradictions.

#### Implemented comparison baseline

The first association baseline is implemented in `fact3r/association/`. It:

1. gates proposal-entity pairs by expanded 3D bounding-box overlap or centroid distance;
2. constructs one reusable cost matrix from centroid distance, padded 3D box overlap,
   symmetric point consistency, robust RGB statistics and optional pooled MASt3R
   descriptors;
3. renormalizes the active weights when optional colour or descriptor evidence is
   unavailable;
4. applies exact one-to-one Hungarian assignment with a maximum accepted cost; and
5. returns matched pairs plus unmatched proposal and entity indices.

`HungarianEntityMapper` applies this solver once to the complete proposal set for
each keyframe. It updates matched entity geometry, retains unobserved entities and
creates a provisional entity for every unmatched mask. This immediate update rule is
only the hard baseline; it intentionally performs no confirmation transition,
delayed belief, confidence gate or split/merge operation. The runnable boundary is:

```bash
python scripts/run_hungarian_baseline.py \
  --proposals /path/to/fact3r_sam2/scene \
  --output /path/to/fact3r_hungarian/scene
```

The output manifest records frame-level matches and created/unobserved IDs. Each
frame also stores the complete cost matrix, candidate mask and component costs so
later assignment models can be compared against identical evidence.

Every unmatched proposal is assigned one observable diagnostic reason:

- `empty_map`: initialization frame contained no existing entities;
- `no_spatial_candidate`: no entity survived spatial candidate gating;
- `cost_above_threshold`: candidates existed, but even the best cost exceeded the
  configured hard-match threshold;
- `assignment_competition`: a viable candidate existed but was allocated to another
  proposal by the one-to-one solver.

The manifest stores the best candidate entity/cost where one exists, per-frame
reason counts and run-level totals. `no_spatial_candidate` is deliberately not called
"new object": distinguishing new scene content from a gating failure requires
visibility/map-coverage evidence that this baseline does not yet model.

#### Optional SAM2 short-term continuity cue

The official SAM2 video predictor can now propagate every accepted proposal into
the next keyframe. Propagated masks are jointly linked to the next frame's complete
automatic-proposal set by one-to-one 2D mask IoU. Linked proposals inherit a
diagnostic track ID; unlinked proposals start a new track. Each frame is re-anchored
to its automatic masks, so errors do not accumulate through unrestricted video
propagation and the number of live SAM2 objects remains bounded.

```bash
python scripts/build_sam2_tracklets.py \
  --keyframes /path/to/fact3r_keyframes/scene \
  --proposals /path/to/fact3r_sam2/scene \
  --device 0 \
  --min-link-iou 0.30 \
  --max-seeds-per-batch 8

python scripts/run_hungarian_baseline.py \
  --proposals /path/to/fact3r_sam2/scene \
  --tracklets /path/to/fact3r_sam2_tracklets/scene \
  --output /path/to/fact3r_hungarian_tracklets/scene
```

At association time, an adjacent-frame link is resolved through the source
proposal's assigned entity. Its IoU-weighted identity preference is added as a
`temporal` cost component. It cannot bypass the existing 3D spatial gate and does
not force a match; contradictory geometry can still win or leave the proposal
unmatched. Omitting `--tracklets` reproduces the geometry-first Hungarian baseline.

#### Dense one-second HM3D segmentation diagnostic

The rendered HM3D robot trajectory is already a 30 FPS sequence. Sampling one
frame per second would be sparser than the current SLAM keyframes, so the temporal
diagnostic instead selects a one-second window and runs automatic SAM2 on every
captured frame in that window. For example, frames 240–269 include the problematic
staircase view around frame 248:

```bash
conda run -n SAM2 python3 \
  fact3r-map/scripts/run_hm3d_one_second_segmentation.py \
  --sequence datasets/hm3d_seqs/00800-TEEsavR23oF \
  --start-frame 240 \
  --duration-seconds 1 \
  --device 0 \
  --points-per-batch 32
```

Each frame is segmented independently. Accepted masks are linked to the preceding
frame only by one-to-one mask IoU, with no SAM2 video memory, 3D geometry, map
entity, or UOT evidence. Stable track colours in `one_second_tracks.gif` therefore
measure raw automatic-mask consistency under robot motion. The manifest reports
proposal counts, adjacent link rate, median link IoU, track count and track-length
histogram.

This experiment is image-only and cannot feed the 3D mapper directly. Its purpose
is to choose the next controlled change: high short-term stability supports
accumulating tracklet evidence before UOT; rapid track creation means proposal
stability must be improved before tuning the association solver.

#### Balanced Sinkhorn comparison

The next implemented assignment baseline replaces the one-to-one Hungarian solve
with balanced entropic transport while keeping the proposal-entity costs, spatial
candidate mask, match-cost threshold and optional SAM2 temporal cue unchanged.
The active proposal and entity marginals are fixed and uniform. Sinkhorn scaling is
performed in the log domain, and each proposal immediately commits to the viable
entity receiving its largest transport mass. Multiple proposal fragments may
therefore update the same entity in one frame.

```bash
python scripts/run_balanced_sinkhorn.py \
  --proposals /path/to/fact3r_sam2/scene \
  --tracklets /path/to/fact3r_sam2_tracklets/scene \
  --output /path/to/fact3r_balanced_sinkhorn/scene
```

This is deliberately not yet the final association model. There are no dustbins,
unbalanced marginal penalties or delayed commitments. Fixed marginals can force
some soft mass onto high-cost numerical stand-ins for forbidden edges, although
such edges can never become hard matches. The runner reports this as
`mean_forbidden_mass`, along with convergence and marginal error, making the
failure mode explicit for the dustbin and unbalanced comparisons.

#### Visibility-conditioned residual transport

The first unbalanced model is implemented without a shared dustbin row, a shared
dustbin column, or a learned scalar null logit. It is asymmetric because its two
sides have different physical meanings: current SAM2 proposals are observations,
while persistent entities are 3D map state.

Before transport, each entity's stored geometry is projected into the current
MASt3R keyframe. The current depth map separates visible surface from occluded or
out-of-frustum surface. This visibility score sets both the entity's desired mass
and how strongly its marginal is enforced. Proposal mass and marginal strength are
conditioned on mask-to-3D point retention and optional SAM2 tracklet confidence.

Generalized Sinkhorn scaling is then performed only on the original spatial
candidate graph. Non-candidate entries remain `-inf` in the log kernel and exactly
zero in the transport plan. The two directional residuals have explicit meanings:

- proposal mass not transported is **birth/fragment/noise evidence**;
- visible entity mass not supplied is **miss/occlusion evidence**.

```bash
python scripts/run_visibility_residual_transport.py \
  --keyframes /path/to/fact3r_keyframes/scene \
  --proposals /path/to/fact3r_sam2/scene \
  --tracklets /path/to/fact3r_sam2_tracklets/scene \
  --output /path/to/fact3r_visibility_residual_transport/scene
```

The runner saves visibility, desired and transported marginals, birth and miss
residuals, excess mass, the strict-support plan, hard-decision confidence, and
typed rejection reasons. Omitting `--delayed-commitment` preserves immediate
entity creation as the UOT ablation.

#### Tracklet-conditioned delayed commitment

The improved lifecycle keeps the same complete-frame UOT plan, candidate graph,
costs and marginals. Only the interpretation of rejected proposal mass changes.
An unmatched proposal with desired mass `a_i` and birth residual `r_i` contributes
the normalized evidence

```text
rho_i = r_i / a_i
```

to a pending state keyed by its SAM2 track ID. For track `k`, the mapper maintains
the mean `rho`, median adjacent mask IoU and maximum step between globally aligned
3D centroids. A persistent entity is created only when all configured conditions
hold. The defaults require three observations, mean residual ratio at least
`0.55`, median link IoU at least `0.60`, and centroid steps no larger than `0.30`
metres:

```bash
python scripts/run_visibility_residual_transport.py \
  --keyframes /path/to/fact3r_keyframes/scene \
  --proposals /path/to/fact3r_sam2/scene \
  --tracklets /path/to/fact3r_sam2_tracklets/scene \
  --delayed-commitment
```

One-frame pending tracks expire without entering the map. If UOT later rejects an
observation from a track already tied to an entity, the observation is held for
continuity but cannot create a duplicate and does not update entity memory. This
separates delayed birth commitment from the later confidence-gated memory-update
ablation. The default output is written under `fact3r_delayed_commitment_uot` so
the immediate-UOT result is not overwritten.

The manifest records every pending/confirmed/held decision, its accumulated
statistics and blocking reasons, expired tracks, peak pending count, and final
unresolved tracks.

#### Real-rover multi-rate operation

Camera rate and neural-map update rate should not be forced to match. Capture RGB
at 30 FPS into a timestamped bounded queue, let tracking consume the newest frame
at its sustainable rate, propagate existing masks more frequently, and run full
automatic proposal discovery only periodically or on novelty. A practical initial
desktop-GPU schedule is:

| Component | Initial target |
|---|---:|
| Camera capture and recording | 30 FPS |
| MASt3R-SLAM tracking | 10–15 FPS |
| SAM2 video propagation of current tracks | 5–10 FPS |
| Complete-frame automatic SAM2 discovery | 1–2 FPS |
| 3D UOT/entity-map update on finalized keyframes | 2–5 FPS |

These are deployment targets, not claimed end-to-end benchmarks. MASt3R-SLAM
reports 15 FPS and its released implementation notes that experiments used an RTX
4090. Meta reports 39.5 FPS for compiled SAM 2.1 Hiera-L video inference on an
A100; that figure is not the automatic mask generator used here, which evaluates
a dense prompt grid and must be benchmarked separately on the rover computer.
Running both large models on one GPU also introduces contention.

Choose the tracking floor from motion as well as compute. If consecutive mapping
frames should remain within translation `d_max` and rotation `theta_max`, use

```text
tracking_fps >= max(linear_speed / d_max, yaw_rate / theta_max)
```

For example, at `0.5 m/s` with `d_max = 0.04 m`, or at `45 deg/s` with
`theta_max = 3 deg`, the floor is approximately 15 FPS. The capture thread can
remain at 30 FPS while overloaded inference workers drop stale frames rather than
building latency.

References: [MASt3R-SLAM CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html),
[released MASt3R-SLAM implementation](https://github.com/rmurai0610/MASt3R-SLAM),
and [official SAM 2 benchmarks](https://github.com/facebookresearch/sam2#model-description).

#### Image and temporal inspection

Association manifests can be rendered directly over the exported RGB keyframes.
The command accepts multiple mapping runs so the same frame can be compared side
by side:

```bash
python scripts/visualize_association.py \
  --keyframes /path/to/fact3r_keyframes/scene \
  --proposals /path/to/fact3r_sam2/scene \
  --mapping "Hungarian+tracklets=/path/to/fact3r_hungarian_tracklets/scene" \
  --mapping "Balanced Sinkhorn=/path/to/fact3r_balanced_sinkhorn/scene" \
  --mapping "Visibility residual UOT=/path/to/fact3r_visibility_residual_transport/scene" \
  --output /path/to/fact3r_association_visualization/scene
```

Entity colours are stable across frames. Green mask boundaries are matched
observations; red boundaries are newly created entities; yellow boundaries are
pending births coloured by track ID; cyan boundaries are observations held on an
existing track without a memory update. Sinkhorn panels also display convergence,
iteration count and forbidden mass. The output contains every comparison frame as PNG, a sampled
`association_contact_sheet.png`, and an `association.gif` for temporal identity
inspection.

This same `PairwiseCostMatrix` is the input boundary for Hungarian, balanced
Sinkhorn and visibility-conditioned UOT. Keeping the evidence fixed isolates the
effect of the assignment model. A shared learned dustbin remains useful as an
ablation, but is not the central proposed mechanism. The private unmatched columns
used internally by the Hungarian solver are only an implementation mechanism.

### 8.2 Point-level transport score

Within each gated proposal-entity pair, sample proposal points \(p_a\) and entity surfels \(s_b\). Define:

\[
c_{ab} =
\mu_x\|x_a-x_b\|_2^2 +
\mu_d(1-\cos(D_a,D_b)) +
\mu_n(1-n_a^\top n_b) +
\mu_c\Delta_{Lab}(a,b)
\]

Use partial or unbalanced transport to obtain:

- matched mass;
- unmatched mass;
- mean 3D residual;
- mean descriptor residual.

These statistics form \(\rho^{3R}_{ij}\) and \(\rho^{xyz}_{ij}\).

### 8.3 Mask-to-entity unbalanced Sinkhorn

Solve:

\[
P^* =
\arg\min_{P\geq0}
\langle C,P\rangle
-\epsilon H(P)
+\tau_r\mathrm{KL}(P\mathbf 1\|a)
+\tau_c\mathrm{KL}(P^\top\mathbf 1\|b)
\]

The unbalanced formulation permits unmatched or partially matched mass without
adding a shared null row or column. Set proposal evidence $a_i$ from retained 3D
support and temporal confidence. Set entity demand $b_j$ from the fraction of its
3D geometry predicted visible in the current depth map. Use per-node relaxation
strengths so reliable proposals and clearly visible entities are enforced more
strongly than weak masks and occluded entities.

For strict spatial support $S$, generalized Sinkhorn updates are:

\[
K_{ij}=\begin{cases}\exp(-C_{ij}/\epsilon)&(i,j)\in S\\0&\text{otherwise}\end{cases}
\]

\[
u_i=\left(\frac{a_i}{(Kv)_i}\right)^{\tau_i^p/(\tau_i^p+\epsilon)},\qquad
v_j=\left(\frac{b_j}{(K^\top u)_j}\right)^{\tau_j^e/(\tau_j^e+\epsilon)}
\]

This preserves exactly zero mass on impossible pairs. Positive proposal residual
$a-P\mathbf 1$ is evidence for a new object, fragmentation or proposal noise;
positive visible-entity residual $b-P^\top\mathbf 1$ is evidence for a missed or
occluded entity. The direction and cause of null evidence are therefore retained
instead of collapsed into a single learned scalar.

### 8.4 Delayed commitment

Do not immediately convert \(P\) into hard entity IDs. Maintain association beliefs over several observations.

Commit proposal \(m_i\) to entity \(E_j\) only when:

\[
P_{ij} > \tau_{commit}
\]

\[
P_{ij} - \max_{k\ne j}P_{ik} > \Delta_{margin}
\]

and the association is supported by multiple viewpoints or a strong loop-closure match.

Otherwise, retain the proposal as provisional.

### 8.5 Split and merge handling

#### Multiple masks assigned to one entity

Possible causes:

- one object was fragmented into parts;
- a hierarchical mask generator returned both object and part masks.

Action:

- merge masks if repeated evidence shows they form one object;
- otherwise retain them as part entities under the same parent.

#### One mask transports mass to several entities

Possible cause:

- the mask combines multiple physical objects.

Action:

- split its lifted points using point-to-entity transport probabilities;
- re-run mask-to-entity assignment on the resulting 3D components.

#### New or temporarily hidden objects

- high proposal birth residual over multiple observations creates a new entity;
- an unmatched existing entity remains inactive rather than being deleted;
- reactivation requires descriptor and geometric consistency.

## 9. Temporal and self-supervised learning objectives

### 9.1 Association supervision

Use MASt3R and geometry to generate pseudo-labels:

- positives: masks linked by many reciprocal, geometrically consistent matches;
- negatives: simultaneously visible, spatially distinct masks;
- hard negatives: similar-looking entities at different 3D positions;
- long-term positives: re-observations found through loop closure.

### 9.2 Cycle consistency

For mask association matrices across three keyframes:

\[
\mathcal L_{cycle} =
\left\|
P_{t,t+1}P_{t+1,t+2} - P_{t,t+2}
\right\|_1
\]

### 9.3 Geometry consistency

\[
\mathcal L_{geom} =
\sum_{ij}P_{ij}\,
\overline d_{3D}(m_i,E_j)
\]

### 9.4 Assignment supervision

When reliable pseudo-labels or ground-truth instances are available:

\[
\mathcal L_{assoc} = -\sum_{ij}Y_{ij}\log P_{ij}
\]

The initial prototype should use fixed affinity weights. A learned cost network should only be added after the non-learned pipeline is stable.

## 10. Semantic fact extraction

Semantic extraction occurs only after an entity has sufficient geometric stability.

### 10.1 View selection

Choose a small, diverse set of observations using:

- visible entity coverage;
- image sharpness;
- low occlusion;
- distance and scale;
- viewpoint diversity;
- MASt3R confidence.

### 10.2 Structured extraction

Request open-vocabulary facts rather than a single caption:

```json
{
  "type": ["chair"],
  "colour": ["black"],
  "materials": {"seat": "fabric", "legs": "metal"},
  "parts": ["seat", "back", "four legs", "armrests"],
  "state": ["slightly worn"],
  "visible_text": [],
  "affordances": ["sittable"],
  "distinctive_details": ["curved armrests"]
}
```

The semantic model operates on selected views. During development, retain the
exported keyframes so observation queries can render their complete visual
history. A later deployment can replace those frames with compact thumbnails and
source-video offsets after facts have been consolidated.

### 10.3 Multi-view consolidation

For candidate fact \(f\) attached to entity \(E\):

\[
L(f,E) =
\sum_{v\in\mathcal V(E)}
w_v\log\frac{p(f\mid I_v,E)}{1-p(f\mid I_v,E)}
\]

The view weight \(w_v\) incorporates visibility, viewpoint diversity, MASt3R confidence and observation quality.

Conflicting facts are retained as a distribution rather than overwritten. For example:

```text
colour(entity_42):
    black      0.87
    dark brown 0.11
    unknown    0.02
```

### 10.4 Point and part grounding

Each fact should refer to:

- the entire entity;
- a persistent part entity;
- a local 3D box;
- or a sparse subset of supporting points.

This prevents attribute-binding errors such as attaching the colour of a nearby wall to a chair.

### 10.5 SigLIP observation memory

Before structured fact extraction, every saved SAM object or part proposal can be
encoded as a masked, context-preserving SigLIP2 crop. The observation index stores
the embedding together with its proposal ID, frame, timestamp, mask, track and
persistent entity. Delayed observations are retrospectively attached to the
entity eventually confirmed for their track, so the entity history includes the
views seen before commitment.

Build the index from the already-computed keyframes, proposals and delayed-UOT
map; this does not rerun MASt3R-SLAM or SAM2:

```bash
conda run -n SAM2 python3 \
  fact3r-map/scripts/build_siglip_observation_index.py \
  --keyframes logs/hm3d/calib_fact3r/fact3r_keyframes/00800-TEEsavR23oF \
  --proposals logs/hm3d/calib_fact3r/fact3r_sam2/00800-TEEsavR23oF \
  --mapping logs/hm3d/calib_fact3r/fact3r_delayed_commitment_uot/00800-TEEsavR23oF \
  --device 0
```

The manifest records model loading, image-encoding and total indexing time, plus
the measured mask throughput on the actual GPU. Query the index with free text:

```bash
conda run -n SAM2 python3 \
  fact3r-map/scripts/query_siglip_observations.py \
  --index logs/hm3d/calib_fact3r/fact3r_siglip_observations/00800-TEEsavR23oF \
  --query "a clock" \
  --confounder "a ceiling fan" \
  --confounder "a ceiling light" \
  --confounder "a picture frame" \
  --confounder "a smoke detector" \
  --max-entities 3 \
  --device 0
```

Navigation queries rank confirmed entities by default; pending and isolated masks
cannot become return goals. Each view is scored by its positive-prompt similarity
minus its strongest confounder similarity, then weighted by SAM confidence,
association confidence and mask resolution. An entity must have at least two
supporting views and pass both view- and entity-margin gates. Consequently,
`--max-entities` is only an upper bound: the query can return no confident match.
After an entity passes, every indexed observation belonging to it is rendered,
not only its highest-scoring crop. The output contains `results.json`, an HTML
gallery, highlighted frames and, when a match exists, a contact sheet and
`matches.gif`. Use `--include-unconfirmed` only for diagnostic recall inspection,
never for selecting a navigation target.

#### Automatic Qwen3-VL verification (no hand-written confounders)

For open-ended robot queries, use SigLIP as a high-recall shortlist and
Qwen3-VL as a multi-view verifier. Install the extra dependency once:

```bash
conda run -n SAM2 pip install -e 'fact3r-map[vlm]'
```

Then query the existing observation index:

```bash
conda run -n SAM2 python3 \
  fact3r-map/scripts/query_vlm_verified_observations.py \
  --index logs/hm3d/calib_fact3r/fact3r_siglip_observations/00800-TEEsavR23oF \
  --query "a clock" \
  --siglip-device 0 \
  --vlm-model Qwen/Qwen3-VL-8B-Instruct \
  --vlm-device-map auto \
  --max-candidates 6 \
  --evidence-views 3 \
  --min-vlm-confidence 0.75 \
  --min-vlm-supporting-views 2
```

This path does not ask for `--confounder` arguments. SigLIP first ranks only
confirmed persistent entities using the positive query ensemble. It is unloaded
before Qwen3-VL is loaded. For each shortlisted entity, Qwen receives up to three
images; every image pairs the full frame with an enlarged crop and highlights the
candidate mask in green. The VLM must return a structured `yes`, `no`, or
`uncertain` verdict, confidence, supporting frame IDs, its predicted object and
visually confusable labels. Acceptance requires `yes`, the confidence threshold,
and at least two valid supporting views. Invalid or ambiguous model output fails
closed.

The output directory includes the candidate evidence, accepted/rejected verdicts,
automatically discovered dynamic confounders, `results.json`, an HTML gallery and
all stored observations of every accepted entity. VLM verdicts are cached under
the observation index, so an identical model/query/entity/evidence request does
not load or invoke Qwen again. Use `--force-reverify` only when a fresh verdict is
needed. `--attention-implementation flash_attention_2 --vlm-dtype bfloat16` can
be added on a compatible CUDA installation.

## 11. Semantic retrieval

Convert a free-form query into a graph pattern.

Example:

```text
Query: "the black chair with metal legs beside the pink door"

Constraints:
    type(E) approximately chair
    colour(E) approximately black
    material(legs(E)) approximately metal
    beside(E, D)
    type(D) approximately door
    colour(D) approximately pink
```

The initial observation index provides direct image-text retrieval. Structured
facts will later rerank these candidates using:

1. structured field matching;
2. synonym and predicate normalization;
3. text-to-text embeddings for free-form fact values;
4. exact 3D computation for spatial relations;
5. confidence-aware graph reranking.

All query constraints must bind to the same entity or an explicitly connected part. Once an entity is selected, its best observation pose becomes the navigation target in the BEV.

## 12. Proposed repository structure

```text
fact3r-map/
├── README.md
├── configs/
│   ├── mapping.yaml
│   ├── association.yaml
│   ├── semantics.yaml
│   └── evaluation.yaml
├── fact3r/
│   ├── reconstruction/
│   │   ├── mast3r_backend.py
│   │   ├── keyframes.py
│   │   └── pointmap_adapter.py
│   ├── proposals/
│   │   ├── mask_generator.py
│   │   ├── mask_filter.py
│   │   └── lift_to_3d.py
│   ├── entities/
│   │   ├── entity.py
│   │   ├── entity_map.py
│   │   ├── surfel_fusion.py
│   │   └── split_merge.py
│   ├── association/
│   │   ├── gating.py
│   │   ├── pairwise_cost.py
│   │   ├── hungarian.py
│   │   ├── sinkhorn.py
│   │   ├── unbalanced_ot.py
│   │   └── temporal_belief.py
│   ├── semantics/
│   │   ├── view_selection.py
│   │   ├── fact_extractor.py
│   │   ├── fact_fusion.py
│   │   └── fact_graph.py
│   ├── navigation/
│   │   ├── bev_map.py
│   │   ├── topology.py
│   │   └── goal_pose.py
│   ├── query/
│   │   ├── parser.py
│   │   ├── graph_matcher.py
│   │   └── reranker.py
│   └── evaluation/
│       ├── entity_metrics.py
│       ├── semantic_metrics.py
│       └── navigation_metrics.py
├── scripts/
│   ├── build_map.py
│   ├── query_map.py
│   ├── evaluate_association.py
│   └── visualize_entity_map.py
└── tests/
```

## 13. Implementation plan

### Milestone 0 — Freeze the interface

- Define keyframe, lifted proposal, entity and semantic fact data structures.
- Save a small deterministic sequence for regression testing.
- Visualize pointmaps, masks and lifted proposals in the same coordinate frame.

**Exit condition:** a mask selected in a frame appears at the correct location in the global 3D reconstruction.

### Milestone 1 — Entity mapping without learning

- Generate masks for keyframes.
- Lift them into 3D.
- Implement spatial candidate gating.
- Implement reciprocal MASt3R match scoring.
- Implement greedy and Hungarian assignment.
- Add provisional and confirmed entity states.
- Visualize persistent entity colours/IDs across frames.

**Exit condition:** common static objects retain consistent IDs over a short sequence.

### Milestone 2 — Split/merge robustness

- Detect one-mask-to-many-entity assignments.
- Split mixed masks at point level.
- Merge repeated mask fragments or create part hierarchies.
- Add inactive-entity reactivation after occlusion.

**Exit condition:** chairs, tables and doors do not multiply uncontrollably as viewpoint changes.

### Milestone 3 — Unbalanced transport

- Implement balanced Sinkhorn.
- Keep a shared-dustbin model as a controlled ablation.
- Implement visibility-conditioned unbalanced marginal penalties.
- Add delayed hard commitment.
- Compare against Hungarian using identical pairwise costs.

**Exit condition:** UOT reduces identity switches and fragmentation without causing excessive entity merging.

### Milestone 4 — Semantic facts

- Select diverse observations per confirmed entity.
- Extract structured object and part facts.
- Fuse multi-view evidence.
- Ground facts to entity or part support.
- Implement semantic fact inspection and correction tools.

**Exit condition:** “black chair,” “pink door” and compositional attribute queries retrieve the correct entity more reliably than a dense CLIP-map baseline.

### Milestone 5 — Long-horizon retrieval and return

- Connect entities to BEV cells and topology nodes.
- Generate visibility-aware goal poses.
- Add query parsing and graph matching.
- Evaluate after long temporal and spatial gaps.

**Exit condition:** the agent can retrieve and plan back to previously observed entities after long trajectories.

## 14. Baselines and ablations

### Association baselines

1. greedy nearest 3D entity;
2. 3D IoU plus Hungarian;
3. MASt3R descriptor score plus Hungarian;
4. balanced Sinkhorn;
5. Sinkhorn with dustbins;
6. unbalanced Sinkhorn;
7. unbalanced Sinkhorn plus temporal cycle consistency.

### Feature ablations

- without MASt3R descriptors;
- without geometric overlap;
- without confidence weighting;
- without colour/shape cues;
- without delayed commitment;
- without split/merge correction;
- without multi-view semantic consolidation;
- one caption per entity versus grounded propositions.

### Semantic-map baselines

- dense or BEV visual-language feature averaging;
- object-level feature pooling;
- object caption graph;
- proposed point-grounded fact graph.

## 15. Evaluation metrics

### Entity-map quality

- entity association precision and recall;
- IDF1 or equivalent persistent-ID metric;
- identity switches;
- entity fragmentation rate;
- incorrect entity merge rate;
- point-level entity purity and completeness;
- re-identification after long occlusion or loop closure.

### Semantic quality

- object retrieval Recall@1 and Recall@5;
- attribute retrieval accuracy;
- compositional retrieval accuracy;
- part-attribute binding accuracy;
- semantic contradiction rate;
- expected calibration error of fact confidence.

### Long-horizon navigation

- semantic target success rate;
- success weighted by path length;
- final distance to target entity;
- success versus travelled distance;
- success versus revisit delay;
- correct-instance return rate when several similar objects exist.

### Efficiency

- mapping time per keyframe;
- association time per proposal;
- peak GPU memory;
- stored bytes per metre travelled;
- stored bytes per confirmed entity;
- semantic-query latency.

## 16. Dataset strategy

Use a staged evaluation rather than beginning with kilometre-scale deployment.

1. **Controlled short sequences:** validate lifting, association and visualization.
2. **Indoor RGB-D datasets used as evaluation only:** process RGB through the proposed system and use depth/instance annotations only for measurement.
3. **Repeated-object sequences:** deliberately include several similar chairs, doors or cabinets.
4. **Loop-closure sequences:** leave a room and return after a long trajectory.
5. **Long-horizon composed environments:** connect rooms/floors or collect a real long traversal.

The long-horizon benchmark must include instance-specific and attribute-specific queries, not only category queries.

## 17. Main risks

### Mask fragmentation dominates the system

Mitigation:

- begin with high-confidence object-level proposals;
- delay part hierarchy construction;
- use 3D split/merge diagnostics;
- evaluate mask quality separately from association quality.

### MASt3R drift corrupts entity positions

Mitigation:

- use loop-closure-corrected poses;
- version entity coordinates after global optimization;
- avoid irreversible fusion before pose stabilization;
- retain local keyframe coordinates for re-fusion when necessary.

### Sinkhorn produces smooth but incorrect assignments

Mitigation:

- hard geometric gating before transport;
- reduce entropy temperature as evidence grows;
- require assignment margin and temporal persistence;
- retain an unmatched option;
- report incorrect-merge rate, not only association recall.

### Semantic facts remain noisy

Mitigation:

- extract semantics only for geometrically confirmed entities;
- use multiple diverse views;
- store conflicting hypotheses rather than overwriting;
- distinguish measured facts from model-inferred facts;
- calibrate confidence on held-out scenes.

### “Arbitrary query” is interpreted too strongly

The map can answer arbitrary queries about facts that were visually observable and extracted. If all images and uncommitted visual information are deleted, an unrecorded detail cannot be recovered later. This limitation must be stated explicitly.

## 18. Minimum viable experiment

The first experiment should be intentionally small:

1. record or select a sequence containing three chairs, including one black chair;
2. leave the room and later return;
3. run MASt3R-SLAM and export keyframe pointmaps/descriptors;
4. generate class-agnostic masks;
5. create persistent entity IDs using Hungarian assignment;
6. replace Hungarian with UOT using the same costs;
7. attach multi-view facts to confirmed entities;
8. query “black chair” after the return loop;
9. compare entity ID stability and correct-instance retrieval.

This experiment tests the complete research hypothesis without requiring a kilometre-scale system first.

## 19. Success criteria for the first paper-quality version

The approach is promising if it demonstrates all of the following:

- fewer identity switches than Hungarian under mask fragmentation and re-observation;
- lower incorrect-merge rate than balanced Sinkhorn;
- improved compositional semantic retrieval over dense feature averaging;
- correct-instance return when several objects share the same category;
- bounded map growth with distance;
- interpretable evidence showing which views and points support each semantic fact.

## 20. Suggested project name and paper title

Repository name:

```text
fact3r-map
```

Working paper title:

> **Fact3R-Map: Point-Grounded Semantic Mapping with MASt3R-Supervised Unbalanced Entity Association**

Alternative title emphasizing the navigation task:

> **Persistent 3D Entity and Proposition Memory for Long-Horizon Vision-and-Language Navigation**

## 21. Immediate next actions

- [x] Set up MASt3R-SLAM inference and export keyframe pointmaps, poses and confidence.
- [x] Define the `LiftedProposal`, `Entity` and `SemanticFact` schemas.
- [x] Visualize 2D masks lifted into the global reconstruction.
- [ ] Add greedy association as an optional ablation.
- [x] Implement geometric gating and Hungarian assignment.
- [ ] Create a short sequence containing repeated object categories.
- [x] Add entity-ID visualization and assignment diagnostics.
- [x] Implement balanced Sinkhorn and measure forbidden transport mass.
- [x] Add strict-support, visibility-conditioned unbalanced transport.
- [ ] Add a shared-dustbin ablation for comparison only.
- [ ] Add delayed commitment and split/merge handling.
- [ ] Only then add structured multi-view semantic extraction.

---

The recommended development order is deliberate: first make persistent physical identity reliable, then add semantics. A sophisticated semantic representation cannot recover from an entity map that repeatedly merges different objects or assigns new identities whenever the viewpoint changes.
