"""vm-extract-struct -- Extract features from segmented structure-borne HDF5 files.

Usage::

    vm-extract-struct --segments-dir data/raw_data_extracted_splits/structure \
                      --config configs/structure.yaml \
                      --out-csv data/features/structure/features.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from vm_micro.features.structure import extract_structure, resolve_effective_structure_config
from vm_micro.utils import apply_overrides, get_logger, load_config

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vm-extract-struct", description="Extract structure features.")
    p.add_argument("--segments-dir", required=True)
    p.add_argument("--config", default="configs/structure.yaml")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--file-glob", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--extractor", default=None, choices=["v1", "v2", "extensive"])
    p.add_argument("override", nargs="*")
    return p


def _sampling_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in ("sr_hz_native", "sr_hz_used", "ds_rate", "duration_s"):
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            stats[col] = {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
            }
    return stats


def main() -> None:
    args = build_parser().parse_args()
    cfg_all = load_config(args.config)
    cfg = (
        cfg_all.get("classical", cfg_all) if isinstance(cfg_all.get("classical"), dict) else cfg_all
    )
    if args.override:
        cfg = apply_overrides(cfg, args.override)
    if args.extractor is not None:
        cfg["extractor"] = args.extractor

    out_csv = Path(args.out_csv or "data/features/structure/features.csv")
    df = extract_structure(
        segments_dir=args.segments_dir,
        cfg=cfg,
        out_csv=out_csv,
        file_glob=args.file_glob,
        n_workers=args.workers,
    )

    sidecar = Path(str(out_csv) + ".extractor_config.json")
    payload = {
        "schema_version": 1,
        "effective_extraction_config": resolve_effective_structure_config(cfg),
        "source_config_path": str(args.config),
        "cli_overrides": list(args.override or []),
        "rows": len(df),
        "sampling_stats": _sampling_stats(df),
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Extracted {len(df)} rows  {len(df.columns)} columns  {out_csv}")


if __name__ == "__main__":
    main()
