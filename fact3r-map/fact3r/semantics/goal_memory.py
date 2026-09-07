"""Cross-session recall for "go back to X" queries against one semantic map.

A resolved query is cheap to redo but not free -- the shortlist-then-verify
flow in `resolve_semantic_goal_verified.py` loads two models and renders
photo evidence every time. When an operator asks for the same object again,
possibly phrased as "go back to the mug" instead of "the mug", the answer
should come from what was already found, not a fresh SigLIP+Qwen pass.

The record lives next to the map manifest (`{stem}_goal_memory.jsonl`), not
inside a `--run` directory, because it is a property of the map, not of one
pipeline invocation: it must survive across runs and across sessions.

No embedding is persisted to disk. Two paraphrases of the same object can
score anywhere from very high to unexpectedly mediocre similarity under
SigLIP2's text tower (it is trained for text<->image alignment, not
text<->text), so a single stored-embedding threshold would be a bigger
unknown to tune blind than re-embedding the handful of distinct past queries
on demand -- there are never more than a few dozen -- with whatever encoder
is already loaded for this stage's own retrieval ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


_NAVIGATION_PREFIXES = (
    "go back to",
    "return to",
    "take me to",
    "navigate to",
    "drive to",
    "go to",
    "head to",
    "back to",
)


class TextEncoder(Protocol):
    def encode_text(self, texts: Sequence[str]) -> NDArray[np.floating]: ...


def normalize_query(text: str) -> str:
    """Casefold, collapse whitespace, and strip a leading navigation command.

    Generic command-shape normalization, not object-specific rule-writing:
    "go back to the mug" and "the mug" should compare equal without either
    query needing to name what kind of object it is.
    """

    cleaned = " ".join(text.strip().casefold().split())
    for prefix in _NAVIGATION_PREFIXES:
        if cleaned.startswith(prefix + " "):
            cleaned = cleaned[len(prefix) + 1 :].strip()
            break
    return cleaned


def load_memory(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def append_memory(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def find_memory_hit(
    query: str,
    records: Sequence[dict[str, object]],
    encoder: TextEncoder,
    *,
    min_similarity: float = 0.90,
) -> dict[str, object] | None:
    """The most recent past record this query most plausibly refers to.

    Tier 0 (exact, no model call): `normalize_query` equality -- handles the
    dominant case of the same object phrase, with or without a "go back"
    prefix, for free. Tier 1 (fallback): SigLIP text-embedding cosine
    similarity against the distinct set of past query texts, using whatever
    encoder this stage already has resident.
    """

    if not records:
        return None
    normalized_query = normalize_query(query)
    for record in reversed(records):
        if normalize_query(str(record["query_text"])) == normalized_query:
            print(f'memory: exact match on {record["query_text"]!r}')
            return record

    distinct_texts = list(dict.fromkeys(str(r["query_text"]) for r in records))
    vectors = np.asarray(
        encoder.encode_text([query, *distinct_texts]), dtype=np.float32
    )
    similarities = vectors[1:] @ vectors[0]
    best_index = int(np.argmax(similarities))
    best_similarity = float(similarities[best_index])
    print(
        f"memory: best paraphrase similarity {best_similarity:.3f} "
        f"(threshold {min_similarity:.3f}) on {distinct_texts[best_index]!r}"
    )
    if best_similarity < min_similarity:
        return None
    best_text = distinct_texts[best_index]
    for record in reversed(records):
        if str(record["query_text"]) == best_text:
            return record
    return None
