# Fact3R real-video ablation study

This protocol isolates the contribution of each association cue and the semantic
embedding backend while reusing exactly the same sampled frames and SAM2 masks.
It therefore does not confuse a change in segmentation recall with a change in
identity association.

## Controlled variants

| Variant | Appearance | SAM2 temporal link | MASt3R correspondence | Semantic model |
| --- | --- | --- | --- | --- |
| SigLIP-A | yes | no | no | SigLIP2 |
| SigLIP-A+S | yes | yes | no | SigLIP2 |
| SigLIP-A+M | yes | no | yes | SigLIP2 |
| SigLIP-A+S+M | yes | yes | yes | SigLIP2 |
| Qwen-A+S+M | yes | yes | yes | Qwen3-VL-Embedding |

Here `A` is mask-crop appearance, `S` is the causal SAM2/optical-flow tracklet
link, and `M` is MASt3R reciprocal keypoint support. Every variant uses the same
unbalanced optimal-transport solver and thresholds. The Qwen row is included
only when the completed run contains `qwen_pre_uot/manifest.json`.

Run the label-free association study after the full video has both a SigLIP
pre-UOT index and the shared proposals, tracklets, and MASt3R matches:

```bash
bash scripts/run_fact3r_ablation.sh \
  --run-root logs/fact3r_real_uot/full_video_qwen_complete
```

The command writes:

```text
logs/fact3r_real_uot/full_video_qwen_complete/ablation/report/report.md
logs/fact3r_real_uot/full_video_qwen_complete/ablation/report/metrics.csv
logs/fact3r_real_uot/full_video_qwen_complete/ablation/report/metrics.json
```

## Association metrics

For (N) assigned mask observations, (M) transported matches, and (B) new
entity births, the report includes

\[
r_{\mathrm{match}}=\frac{M}{N}, \qquad
r_{\mathrm{birth}}=\frac{B}{N}.
\]

For each SAM2 tracklet (t), let (E_t) be the ordered entity identities to
which its observations were assigned. Track fragmentation is

\[
F=\frac{1}{|T|}\sum_{t\in T}\mathbf{1}\bigl(|\operatorname{unique}(E_t)|>1\bigr),
\]

and the switch rate is the number of identity changes divided by the number of
temporal tracklet transitions. Lower values indicate better short-term identity
continuity.

For normalized semantic embeddings (x_i) assigned to entity (e), with
normalized entity prototype

\[
\mu_e=\frac{\sum_{i\in e}x_i}{\left\|\sum_{i\in e}x_i\right\|_2},
\]

within-entity coherence is the mean (x_i^\top\mu_e) over multi-view entities.
High coherence indicates that an entity has not accumulated visually unrelated
masks. The median nearest-entity cosine exposes duplicate identities: a high
value means that multiple entities have nearly identical semantic prototypes.

These diagnostics cannot alone establish semantic correctness. A low birth rate
may result from desirable duplicate removal or from harmful over-merging.

## Annotated semantic retrieval

For defensible retrieval results, annotate a few target points in several
frames. A point is sufficient because the stored SAM mask determines which
entity covers that target. Use at least three separated views per physical
object when possible.

Example `annotations.json` (replace the frame IDs and coordinates with real
values from the video):

```json
{
  "queries": [
    {
      "query": "3D printer",
      "targets": [
        {
          "target_id": "printer-1",
          "points": [
            {"frame_id": 120, "xy": [640, 360]},
            {"frame_id": 180, "xy": [590, 350]}
          ]
        }
      ]
    },
    {
      "query": "office chair",
      "targets": [
        {
          "target_id": "chair-left",
          "points": [
            {"frame_id": 40, "xy": [180, 430]}
          ]
        }
      ]
    }
  ]
}
```

Compare all association variants and both semantic backends with the same
annotations:

```bash
conda run --no-capture-output -n SAM2 python3 \
  fact3r-map/scripts/evaluate_semantic_retrieval_ablation.py \
  --variant SigLIP-A=logs/fact3r_real_uot/full_video_qwen_complete/ablation/siglip_appearance \
  --variant SigLIP-A+S=logs/fact3r_real_uot/full_video_qwen_complete/ablation/siglip_sam2 \
  --variant SigLIP-A+M=logs/fact3r_real_uot/full_video_qwen_complete/ablation/siglip_mast3r \
  --variant SigLIP-A+S+M=logs/fact3r_real_uot/full_video_qwen_complete/ablation/siglip_full \
  --variant Qwen-A+S+M=logs/fact3r_real_uot/full_video_qwen_complete/ablation/qwen_full \
  --annotations annotations.json \
  --output logs/fact3r_real_uot/full_video_qwen_complete/ablation/retrieval \
  --device 0 \
  --device-map auto \
  --dtype bfloat16
```

The evaluator reports mean reciprocal rank and Recall@1/5/10 for three entity
aggregation rules:

- `prototype`: long-term entity mean;
- `best_view`: strongest individual observation;
- `hybrid`: equal average of prototype and best-view scores.

This separates two questions: which cues preserve identity best, and which
semantic backend/aggregation retrieves the correct physical object best.
