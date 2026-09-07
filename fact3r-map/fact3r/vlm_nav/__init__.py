"""Pipeline B: end-to-end VLM goal recall, parallel to the SigLIP-based one.

Nothing in `fact3r.semantics` or `fact3r.experiments` is imported-and-changed
here -- only imported-and-reused. A* (`thirdparty/safediffuser`), the
follower, and the scoring code stay exactly what Pipeline A already uses. The
only thing this package replaces is how a text query becomes a goal: Qwen
looks at the rendered semantic BEV and points at a pixel, instead of SigLIP
shortlisting candidates.

See `bev_pointing.py` and `scripts/resolve_semantic_goal_vlm.py`.
"""
