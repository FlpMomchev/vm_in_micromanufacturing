"""vm-train-cls -- Train classical ML models on selected features.

Usage::

    vm-train-cls --features-csv data/features/airborne/features_selected.csv \
                 --out-dir models/features/air/final_models_fast_top3 \
                 --preset balanced --ensemble-top-n 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from vm_micro.classical.trainer import AVAILABLE_MODEL_NAMES, train_classical
from vm_micro.utils import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-train-cls",
        description="Train classical ML models with grouped splits and nested CV.",
    )
    p.add_argument("--features-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--holdout-runs",
        nargs="*",
        default=None,
        help="recording_root values for external holdout only.",
    )
    p.add_argument("--external-holdout-csv", default=None)
    p.add_argument("--train-fraction", type=float, default=None)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--preset", default="balanced", choices=["fast", "balanced", "exhaustive"])
    p.add_argument("--n-iter", type=int, default=160)
    p.add_argument("--search-n-jobs", type=int, default=-1)
    p.add_argument("--outer-max-splits", type=int, default=7)
    p.add_argument("--inner-max-splits", type=int, default=5)
    p.add_argument("--model", nargs="+", default=None, choices=AVAILABLE_MODEL_NAMES)
    p.add_argument("--include-gpr", action="store_true")
    p.add_argument("--skip-slow", action="store_true")
    p.add_argument("--use-cuda", action="store_true")
    p.add_argument("--ensemble-top-n", type=int, default=1)
    p.add_argument("--snap-predictions", action="store_true")
    p.add_argument("--target-mae", type=float, default=0.05)
    p.add_argument("--doe-step", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    logger.info(
        "vm-train-cls | preset=%s | use_cuda=%s | ensemble_top_n=%d",
        args.preset,
        args.use_cuda,
        args.ensemble_top_n,
    )

    result = train_classical(
        features_csv=args.features_csv,
        out_dir=args.out_dir,
        holdout_runs=args.holdout_runs,
        external_holdout_csv=args.external_holdout_csv,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        outer_max_splits=args.outer_max_splits,
        inner_max_splits=args.inner_max_splits,
        preset=args.preset,
        n_iter=args.n_iter,
        search_n_jobs=args.search_n_jobs,
        requested_models=args.model,
        include_gpr=args.include_gpr,
        skip_slow_models=args.skip_slow,
        use_cuda=args.use_cuda,
        ensemble_top_n=args.ensemble_top_n,
        snap_predictions=args.snap_predictions,
        target_mae=args.target_mae,
        doe_step=args.doe_step,
        random_state=args.seed,
    )

    val = result["validation_metrics"]
    test = result["test_metrics"]
    print("\n=== Training complete ===")
    print(f"Best model         : {result['best_model_name']}")
    print(f"Validation MAE     : {val['holdout_mae']:.4f} mm")
    print(f"Internal test MAE  : {test['holdout_mae']:.4f} mm")
    ext = result.get("external_holdout_metrics")
    if ext:
        print(f"External holdout   : {ext['holdout_mae']:.4f} mm")
    print(f"Uncertainty        : {result['total_uncertainty']:.4f} mm")
    if result.get("ensemble_metrics"):
        ens = result["ensemble_metrics"]
        print(f"Ensemble (top-{ens['ensemble_n_members']})   : MAE={ens['ensemble_mae']:.4f} mm")
    print(f"Bundle             : {result['bundle_path']}")


if __name__ == "__main__":
    main()
