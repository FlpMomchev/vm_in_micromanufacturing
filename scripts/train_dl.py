"""vm-train-dl -- Train the deep-learning depth prediction model.

Config-driven: hyperparameters live in the modality config (``dl`` section).

Usage::

    vm-train-dl --data-dir data/raw_data_extracted_splits/air/live \
                --output-dir models/dl/air/reg/my_run --task regression

    vm-train-dl --data-dir ... --output-dir ... --final-only
    vm-train-dl --data-dir ... --output-dir ... epochs=30 lr=5e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import torch

from vm_micro.dl.config import TrainConfig
from vm_micro.dl.training import (
    choose_final_training_epochs,
    fit_final_model_all_files,
    fit_repeated_experiment,
    make_main_split_builder,
)
from vm_micro.dl.utils import (
    add_class_labels,
    attach_step_idx_if_possible,
    build_file_table,
    choose_device,
    dump_json,
    write_label_mapping,
)
from vm_micro.utils import apply_overrides, get_logger, load_config

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-train-dl",
        description="Train the DL depth model (config-driven).",
    )
    p.add_argument("--data-dir", required=True, help="Root of segmented audio files.")
    p.add_argument("--output-dir", required=True, help="Output directory.")
    p.add_argument("--file-glob", default=None, help="Audio file glob override.")
    p.add_argument("--task", choices=["classification", "regression"], default=None)
    p.add_argument("--feature-type", choices=["logmel", "cwt", "linear_spec"], default=None)
    p.add_argument(
        "--model-type",
        choices=["small_cnn", "spec_resnet", "hybrid_spec_transformer"],
        default=None,
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument(
        "--exclude-runs", nargs="*", default=None, help="recording_root values to exclude."
    )
    p.add_argument("--config", default=None, help="Modality config path.")
    p.add_argument("--modality", choices=["airborne", "structure"], default=None)
    p.add_argument("--skip-final-model", action="store_true")
    p.add_argument("--final-only", action="store_true")
    p.add_argument("override", nargs="*", help="Key=value config overrides.")
    return p


def _auto_file_glob(data_dir: str) -> str:
    if any(Path(data_dir).rglob("*.h5")):
        return "**/*.h5"
    return "**/*.flac"


def _infer_modality(data_dir: str) -> str | None:
    tokens = [t for t in Path(data_dir).as_posix().lower().split("/") if t]
    if any(t in {"air", "airborne"} for t in tokens):
        return "airborne"
    if any(t in {"structure", "struct"} for t in tokens):
        return "structure"
    return None


_MODALITY_CONFIGS = {"airborne": "configs/airborne.yaml", "structure": "configs/structure.yaml"}


def _resolve_dl_section(cfg_raw: dict[str, Any], config_path: str) -> dict[str, Any]:
    if "dl" in cfg_raw:
        dl = cfg_raw["dl"]
        if not isinstance(dl, dict):
            raise TypeError(f"Invalid 'dl' section in {config_path}")
        return dl
    if "classical" in cfg_raw:
        raise ValueError(f"{config_path} contains 'classical' but no 'dl' section.")
    return cfg_raw


def _build_cfg(args: argparse.Namespace) -> TrainConfig:
    modality = args.modality or _infer_modality(args.data_dir)
    config_path = args.config or _MODALITY_CONFIGS.get(modality, None)
    if config_path is None:
        raise ValueError("Could not infer modality. Pass --modality or --config.")

    logger.info("Using DL config from %s", config_path)
    cfg_dict = _resolve_dl_section(load_config(config_path), config_path)

    for attr in ("task", "feature_type", "model_type", "device"):
        val = getattr(args, attr, None)
        if val is not None:
            cfg_dict[attr] = val
    cfg_dict["data_dir"] = args.data_dir
    cfg_dict["output_dir"] = args.output_dir

    if args.override:
        cfg_dict = apply_overrides(cfg_dict, args.override)

    if args.file_glob is not None:
        cfg_dict["file_glob"] = args.file_glob
    elif not cfg_dict.get("file_glob"):
        cfg_dict["file_glob"] = _auto_file_glob(args.data_dir)

    valid = TrainConfig.__dataclass_fields__.keys()  # type: ignore[attr-defined]
    return TrainConfig(**{k: v for k, v in cfg_dict.items() if k in valid})


def main() -> None:
    args = build_parser().parse_args()
    cfg = _build_cfg(args)

    device = choose_device(cfg.device)
    logger.info("Device: %s", device)
    if device != "cuda":
        torch.set_num_threads(1)

    file_df = build_file_table(args.data_dir, cfg.file_glob)
    file_df = attach_step_idx_if_possible(file_df)

    if args.exclude_runs:
        before = len(file_df)
        file_df = file_df[~file_df["recording_root"].isin(args.exclude_runs)].copy()
        logger.info("Excluded %d files from runs %s", before - len(file_df), args.exclude_runs)

    file_df, depth_to_class, class_to_depth = add_class_labels(file_df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_label_mapping(class_to_depth, out_dir / "label_mapping.json")
    dump_json(
        {
            "depth_to_class": {str(k): v for k, v in depth_to_class.items()},
            "class_to_depth": {str(k): v for k, v in class_to_depth.items()},
        },
        out_dir / "label_mapping_full.json",
    )

    split_builder = make_main_split_builder(
        split_strategy=cfg.split_strategy,
        evaluation_unit=cfg.evaluation_unit,
        group_mode=cfg.group_mode,
        train_fraction=float(cfg.train_fraction),
        val_fraction=float(cfg.val_fraction),
        test_fraction=float(cfg.test_fraction),
    )

    n_repeats = int(cfg.n_repeats)
    run_final = bool(cfg.run_final_model)
    final_epochs_cfg = int(cfg.final_epochs) if cfg.final_epochs is not None else None

    # Repeated experiment
    experiment = None
    if not args.final_only:
        logger.info(
            "Repeated experiment: task=%s frontend=%s model=%s repeats=%d",
            cfg.task,
            cfg.feature_type,
            cfg.model_type,
            n_repeats,
        )
        experiment = fit_repeated_experiment(
            cfg=cfg,
            file_df=file_df,
            root_out_dir=out_dir,
            n_repeats=n_repeats,
            split_builder=split_builder,
            class_to_depth=class_to_depth,
            device=device,
        )

    # Final model
    if not (args.skip_final_model or not run_final):
        if experiment is not None:
            n_final = choose_final_training_epochs(
                experiment.summary_df,
                explicit_epochs=final_epochs_cfg,
            )
        else:
            if final_epochs_cfg is None:
                raise ValueError("--final-only requires final_epochs in config or override.")
            n_final = final_epochs_cfg

        logger.info("Training final model on ALL files for %d epochs.", n_final)
        fit_final_model_all_files(
            cfg=cfg,
            file_df=file_df,
            root_out_dir=out_dir,
            class_to_depth=class_to_depth,
            device=device,
            final_epochs=n_final,
        )
        logger.info("Final model saved to %s/final_model/", out_dir)

    if experiment is not None and not experiment.summary_df.empty:
        df = experiment.summary_df
        print("\n=== Repeated experiment summary ===")
        for col in [c for c in df.columns if c.startswith(("test_", "val_"))]:
            print(f"  {col:35s}: {df[col].mean():.4f}  {df[col].std():.4f}")


if __name__ == "__main__":
    main()
