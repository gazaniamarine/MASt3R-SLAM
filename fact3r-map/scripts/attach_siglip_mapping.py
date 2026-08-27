#!/usr/bin/env python3
"""Attach final persistent identities to a pre-UOT SigLIP index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.observation_index import (  # noqa: E402
    attach_mapping_to_observation_index,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = attach_mapping_to_observation_index(
        index=args.index,
        mapping=args.mapping,
        output=args.output,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        f"Attached {manifest['assigned_observation_count']}/"
        f"{manifest['observation_count']} observations to persistent entities "
        "without re-encoding images"
    )
    print(f"Wrote mapped SigLIP observation index to {manifest_path}")


if __name__ == "__main__":
    main()
