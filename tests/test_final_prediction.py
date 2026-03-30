"""Tests for scripts.final_prediction helper persistence behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.final_prediction import (
    _attach_actual_depth_mm_to_predictions_csv,
    _bundle_from_pred_csv,
    _canonical_long_record_keys,
    _cleanup_single_model_predictions_csv,
    _copy_split_debug_plots_to_run_dir,
    _persist_model_setup_lock,
    _save_bundle_predictions_csv,
    _single_prediction_report_payload,
    build_parser,
)
from vm_micro.fusion.fuser import PredictionBundle


def _example_setup_audit() -> dict[str, object]:
    return {
        "models": {
            "airborne_dl": {
                "model_kind": "dl",
                "model_dir": "models/dl/air/reg/example",
                "config_path": "models/dl/air/reg/example/final_model/config.json",
                "config_sha256": "abc123",
                "reference_mae_raw_mm": 0.0123,
                "reference_mae_source": "final_model/test_metrics.csv.holdout_mae_raw",
            }
        },
        "run_config_snapshot": {
            "final_prediction_config": "configs/fusion.yaml",
        },
    }


def test_save_bundle_predictions_csv_writes_parent_dirs(tmp_path: Path) -> None:
    bundle = PredictionBundle(
        modality="airborne_ensemble",
        record_names=np.array(["r1", "r2"]),
        y_pred=np.array([0.2, 0.3], dtype=np.float64),
        sigma=np.array([0.01, 0.02], dtype=np.float64),
        validation_mae=0.01,
        y_true=np.array([0.21, 0.29], dtype=np.float64),
    )

    out_csv = tmp_path / "airborne" / "fusion_predictions.csv"
    result = _save_bundle_predictions_csv(bundle, out_csv)

    assert result == out_csv
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert list(df.columns) == [
        "record_name",
        "y_pred",
        "sigma",
        "modality",
        "validation_mae",
        "depth_mm",
        "y_true",
        "residual_mm",
        "abs_residual_mm",
    ]
    assert len(df) == 2


def test_persist_model_setup_lock_writes_lock_files_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260329_120000__manual"
    final_dir = run_dir / "final"
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts, warnings = _persist_model_setup_lock(
        setup_audit=_example_setup_audit(),
        run_dir=run_dir,
        final_dir=final_dir,
    )

    assert warnings == []
    assert set(artifacts.keys()) == {"run_lock", "latest_lock"}

    run_lock = run_dir / artifacts["run_lock"]
    latest_lock = run_dir / artifacts["latest_lock"]
    assert run_lock.exists()
    assert latest_lock.exists()
    assert not list((final_dir / "model_setup_locks").glob("*setup_audit.json"))

    payload = json.loads(run_lock.read_text(encoding="utf-8"))
    assert payload["run_name"] == run_dir.name
    assert payload["final_prediction_config"] == "configs/fusion.yaml"


def test_copy_split_debug_plots_to_run_dir_copies_expected_files(tmp_path: Path) -> None:
    split_out_dir = tmp_path / "splits" / "rec1"
    split_out_dir.mkdir(parents=True, exist_ok=True)
    core_src = split_out_dir / "rec1__debug__core.png"
    padded_src = split_out_dir / "rec1__debug__padded.png"
    core_src.write_bytes(b"core-bytes")
    padded_src.write_bytes(b"padded-bytes")

    run_split_dir = tmp_path / "fusion" / "airborne" / "rec1"
    copied = _copy_split_debug_plots_to_run_dir(
        {"debug_core": str(core_src), "debug_padded": str(padded_src)},
        run_split_dir,
    )

    core_dst = run_split_dir / core_src.name
    padded_dst = run_split_dir / padded_src.name
    assert core_dst.exists()
    assert padded_dst.exists()
    assert core_dst.read_bytes() == b"core-bytes"
    assert padded_dst.read_bytes() == b"padded-bytes"
    assert copied["debug_core"] == str(core_dst)
    assert copied["debug_padded"] == str(padded_dst)


def test_copy_split_debug_plots_to_run_dir_skips_missing_sources(tmp_path: Path) -> None:
    run_split_dir = tmp_path / "fusion" / "structure" / "rec2"
    copied = _copy_split_debug_plots_to_run_dir(
        {
            "debug_core": str(tmp_path / "missing__debug__core.png"),
            "debug_padded": None,
        },
        run_split_dir,
    )
    assert copied == {}


def test_build_parser_parses_mode_and_optional_actual_depth_mm() -> None:
    parser = build_parser()
    args = parser.parse_args(["single", "--actual-depth-mm", "0.875"])
    assert args.mode == "single"
    assert args.actual_depth_mm == 0.875


def test_attach_actual_depth_mm_to_predictions_csv_overwrites_depth_and_residual(
    tmp_path: Path,
) -> None:
    out_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "record_name": ["r1", "r2"],
            "y_pred": [0.65, 0.80],
            "depth_mm": [0.10, 0.10],
            "residual": [-0.55, -0.70],
        }
    ).to_csv(out_csv, index=False)

    _attach_actual_depth_mm_to_predictions_csv(out_csv, 0.75)

    df = pd.read_csv(out_csv)
    assert list(df["depth_mm"]) == [0.75, 0.75]
    assert np.allclose(df["residual"].to_numpy(), np.array([0.10, -0.05]), atol=1e-12)


def test_cleanup_single_model_predictions_csv_drops_legacy_duplicates(tmp_path: Path) -> None:
    out_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "record_name": ["r1", "r2"],
            "y_pred": [0.65, 0.80],
            "depth_mm": [0.75, 0.75],
            "y_true_depth": [0.75, 0.75],
            "residual": [-0.10, 0.05],
        }
    ).to_csv(out_csv, index=False)

    _cleanup_single_model_predictions_csv(out_csv)

    df = pd.read_csv(out_csv)
    assert "residual" not in df.columns
    assert "y_true_depth" not in df.columns
    assert {"y_true", "residual_mm", "abs_residual_mm"}.issubset(df.columns)


def test_single_prediction_report_payload_contains_expected_sections(tmp_path: Path) -> None:
    final_csv = tmp_path / "final" / "final_predictions.csv"
    final_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "record_name": ["rec_1"],
            "y_pred": [0.82],
            "sigma": [0.03],
            "depth_mm": [0.75],
            "residual": [0.07],
        }
    ).to_csv(final_csv, index=False)

    payload = _single_prediction_report_payload(
        run_dir=tmp_path,
        final_predictions_csv=final_csv,
        final_quality={
            "predictions_csv": "final/final_predictions.csv",
            "has_ground_truth": True,
            "n_with_ground_truth": 1,
            "mae_mm": 0.07,
            "rmse_mm": 0.07,
        },
        batch_quality={
            "models": {"airborne_dl": {"mae_mm": 0.08}},
            "modality_fusions": {"airborne_ensemble": {"mae_mm": 0.07}},
        },
        actual_depth_mm=0.75,
    )

    assert payload["mode"] == "single"
    assert payload["actual_depth_mm"] == 0.75
    assert "models" in payload
    assert "modality_fusions" in payload
    assert payload["final_prediction"]["prediction_summary"]["n_predictions"] == 1
    assert payload["final_prediction"]["predictions"][0]["record_name"] == "rec_1"


def test_bundle_from_pred_csv_supports_fusion_and_record_key_modes(tmp_path: Path) -> None:
    pred_csv = tmp_path / "pred.csv"
    pd.DataFrame(
        {
            "record_name": [
                "runA__seg001__step001__A1__depth0.750",
                "runB__seg001__step001__A1__depth0.750",
            ],
            "y_pred": [0.7, 0.8],
            "sigma": [0.01, 0.02],
        }
    ).to_csv(pred_csv, index=False)

    b_fusion = _bundle_from_pred_csv(
        pred_csv,
        "airborne_ensemble",
        fusion_mae=0.02,
        fusion_mae_source="test",
        record_key_mode="fusion",
    )
    b_record = _bundle_from_pred_csv(
        pred_csv,
        "airborne_ensemble",
        fusion_mae=0.02,
        fusion_mae_source="test",
        record_key_mode="record",
    )

    assert b_fusion.metadata["record_key"].startswith("step+hole")
    assert b_record.metadata["record_key"] == "record_name (full)"
    assert b_fusion.record_names.tolist() == ["step=001__hole=A1", "step=001__hole=A1"]
    assert b_record.record_names.tolist() == [
        "runA__seg001__step001__A1__depth0.750",
        "runB__seg001__step001__A1__depth0.750",
    ]


def test_canonical_long_record_keys_are_unique_and_run_indexed() -> None:
    keys = _canonical_long_record_keys(
        np.array(
            [
                "runB__seg001__step001__A1__depth0.750",
                "runA__seg001__step001__A1__depth0.750",
                "runA__seg002__step001__A1__depth0.750",
            ]
        )
    )
    assert len(keys) == 3
    assert len(set(keys.tolist())) == 3
    assert keys[0].startswith("run=002__")
    assert keys[1].startswith("run=001__")
