"""vm_micro.fusion.fuser -- Self-contained fusion layer.

Stage 1 (intra-modality):
    classical + DL  ->  modality ensemble  (weighted by inverse validation MAE)

Stage 2 (inter-modality):
    airborne ensemble + structure ensemble  ->  final prediction

Uncertainty:
    intra  : sigma = sqrt(sum_i w_i * (x_i - mu)^2)           (disagreement)
    inter  : sigma = sqrt(sum_i w_i * ((mu_i - mu)^2 + s_i^2)) (hierarchical)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_EPS = 1e-9


# ---------------------------------------------------------------------------
# PredictionBundle
# ---------------------------------------------------------------------------


@dataclass
class PredictionBundle:
    """Standardised prediction output from any upstream model.

    Parameters
    ----------
    modality        : e.g. ``'airborne_classical'``, ``'airborne_ensemble'``.
    record_names    : 1-D array of segment / file identifiers.
    y_pred          : 1-D float array of depth predictions (mm).
    sigma           : Per-prediction uncertainty (std, mm).
    validation_mae  : Scalar reference MAE used for fusion weighting.
    y_true          : Optional ground-truth labels (mm).
    class_probs     : Optional (N, C) probability matrix.
    metadata        : Free-form dict for diagnostics.
    """

    modality: str
    record_names: np.ndarray
    y_pred: np.ndarray
    sigma: np.ndarray
    validation_mae: float
    y_true: np.ndarray | None = None
    class_probs: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.record_names = np.asarray(self.record_names)
        self.y_pred = np.asarray(self.y_pred, dtype=np.float64)
        self.sigma = np.broadcast_to(
            np.asarray(self.sigma, dtype=np.float64),
            self.y_pred.shape,
        ).copy()
        if self.y_true is not None:
            self.y_true = np.asarray(self.y_true, dtype=np.float64)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "record_name": self.record_names,
                "y_pred": self.y_pred,
                "sigma": self.sigma,
                "modality": self.modality,
                "validation_mae": self.validation_mae,
            }
        )
        if self.y_true is not None:
            df["depth_mm"] = self.y_true
            df["residual"] = self.y_pred - self.y_true
        return df


# ---------------------------------------------------------------------------
# Weighting primitives
# ---------------------------------------------------------------------------


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalise positive weights; fallback to equal weights when invalid."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        return w
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        return np.full(w.shape, 1.0 / w.size, dtype=np.float64)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0:
        return np.full(w.shape, 1.0 / w.size, dtype=np.float64)
    return w / total


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64).ravel()
    w = normalize_weights(weights)
    if vals.shape != w.shape:
        raise ValueError(f"Shape mismatch: values={vals.shape}, weights={w.shape}")
    return float(np.sum(w * vals)) if vals.size else float("nan")


def weighted_sigma(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64).ravel()
    w = normalize_weights(weights)
    if vals.shape != w.shape:
        raise ValueError(f"Shape mismatch: values={vals.shape}, weights={w.shape}")
    if vals.size == 0:
        return float("nan")
    mu = float(np.sum(w * vals))
    return float(np.sqrt(max(float(np.sum(w * (vals - mu) ** 2)), 0.0)))


def hierarchical_sigma(
    means: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray,
) -> float:
    mu = weighted_mean(means, weights)
    m = np.asarray(means, dtype=np.float64).ravel()
    s = np.asarray(sigmas, dtype=np.float64).ravel()
    w = normalize_weights(weights)
    if m.shape != w.shape or s.shape != w.shape:
        raise ValueError(f"Shape mismatch: means={m.shape}, sigmas={s.shape}, weights={w.shape}")
    if m.size == 0:
        return float("nan")
    return float(np.sqrt(max(float(np.sum(w * ((m - mu) ** 2 + s**2))), 0.0)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _inverse_mae_weights(maes: np.ndarray, min_weight: float = 0.05) -> np.ndarray:
    mae_arr = np.asarray(maes, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / (mae_arr + _EPS)
    return normalize_weights(np.maximum(raw, float(min_weight)))


def _align_records(
    bundles: list[PredictionBundle],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Align bundles to the intersection of record names."""
    common = set(bundles[0].record_names.tolist())
    for b in bundles[1:]:
        common &= set(b.record_names.tolist())
    common_records = np.array(sorted(common))

    preds_list: list[np.ndarray] = []
    sigmas_list: list[np.ndarray] = []
    for b in bundles:
        idx = {r: i for i, r in enumerate(b.record_names)}
        ii = [idx[r] for r in common_records]
        preds_list.append(b.y_pred[ii])
        sigmas_list.append(b.sigma[ii])

    return common_records, preds_list, sigmas_list


def _batch_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray | None,
) -> dict[str, Any]:
    if y_true is None:
        return {
            "has_ground_truth": False,
            "n_with_ground_truth": 0,
            "mae_mm": None,
            "rmse_mm": None,
        }
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    n = int(np.sum(mask))
    if n == 0:
        return {
            "has_ground_truth": False,
            "n_with_ground_truth": 0,
            "mae_mm": None,
            "rmse_mm": None,
        }
    res = yp[mask] - yt[mask]
    return {
        "has_ground_truth": True,
        "n_with_ground_truth": n,
        "mae_mm": float(np.mean(np.abs(res))),
        "rmse_mm": float(np.sqrt(np.mean(res**2))),
    }


def bundle_batch_metrics(bundle: PredictionBundle) -> dict[str, Any]:
    """Current-batch metrics (if labels are present)."""
    return _batch_metrics(bundle.y_pred, bundle.y_true)


# ---------------------------------------------------------------------------
# Core fusion
# ---------------------------------------------------------------------------


def _fuse(
    bundles: list[PredictionBundle],
    fused_modality: str,
    min_weight: float = 0.05,
    propagate_internal_sigma: bool = True,
) -> PredictionBundle:
    """Weighted average fusion with disagreement-based uncertainty."""
    if not bundles:
        raise ValueError("Need at least one bundle to fuse.")
    if len(bundles) == 1:
        b = bundles[0]
        return PredictionBundle(
            modality=fused_modality,
            record_names=b.record_names.copy(),
            y_pred=b.y_pred.copy(),
            sigma=b.sigma.copy(),
            validation_mae=b.validation_mae,
            y_true=b.y_true.copy() if b.y_true is not None else None,
            metadata={
                "source_modalities": [b.modality],
                "has_real_sigma": bool(b.metadata.get("has_real_sigma", False)),
            },
        )

    maes = np.array([b.validation_mae for b in bundles])
    weights = _inverse_mae_weights(maes, min_weight=min_weight)
    common, preds_list, sigmas_list = _align_records(bundles)
    n_rows = len(common)
    n_models = len(preds_list)
    preds_mat = np.vstack(preds_list) if n_models else np.zeros((0, n_rows), dtype=np.float64)
    sigmas_mat = np.vstack(sigmas_list) if n_models else np.zeros((0, n_rows), dtype=np.float64)

    y_fused = np.full(n_rows, np.nan, dtype=np.float64)
    sigma_fused = np.full(n_rows, np.nan, dtype=np.float64)
    sigma_between = np.full(n_rows, np.nan, dtype=np.float64)

    for i in range(n_rows):
        row_preds = preds_mat[:, i]
        avail = np.isfinite(row_preds)
        if not np.any(avail):
            continue
        w = normalize_weights(weights[avail])
        means = row_preds[avail]
        y_fused[i] = float(np.sum(w * means))

        bvar = max(float(np.sum(w * (means - y_fused[i]) ** 2)), 0.0)
        sigma_between[i] = float(np.sqrt(bvar))

        if propagate_internal_sigma:
            sigs = sigmas_mat[:, i][avail]
            sigs = np.where(np.isfinite(sigs), sigs, 0.0)
            sigma_fused[i] = hierarchical_sigma(means, sigs, w)
        else:
            sigma_fused[i] = weighted_sigma(means, w)

    # Fused reference MAE
    finite_mask = np.isfinite(maes)
    mae_fused = (
        weighted_mean(maes[finite_mask], normalize_weights(weights[finite_mask]))
        if np.any(finite_mask)
        else float("nan")
    )

    # Propagate ground truth (all bundles must have it)
    y_true_fused: np.ndarray | None = None
    if all(b.y_true is not None for b in bundles):
        _, trues, _ = _align_records(
            [
                PredictionBundle(b.modality, b.record_names, b.y_true, b.sigma, b.validation_mae)  # type: ignore[arg-type]
                for b in bundles
            ]
        )
        y_true_fused = trues[0]

    metadata: dict[str, Any] = {
        "source_modalities": [b.modality for b in bundles],
        "weights": weights.tolist(),
        "source_maes": maes.tolist(),
        "source_batch_metrics": [
            {"modality": b.modality, **bundle_batch_metrics(b)} for b in bundles
        ],
        "has_real_sigma": True,
    }

    if fused_modality == "final_fusion":
        mod_idx = {b.modality: i for i, b in enumerate(bundles)}
        metadata["sigma_between_modalities_mm"] = sigma_between.tolist()
        for name, key in [
            ("airborne_ensemble", "sigma_airborne_mm"),
            ("structure_ensemble", "sigma_structure_mm"),
        ]:
            idx = mod_idx.get(name)
            if idx is not None:
                metadata[key] = np.where(
                    np.isfinite(sigmas_mat[idx]),
                    sigmas_mat[idx],
                    np.nan,
                ).tolist()

    return PredictionBundle(
        modality=fused_modality,
        record_names=common,
        y_pred=y_fused,
        sigma=sigma_fused,
        validation_mae=mae_fused,
        y_true=y_true_fused,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fuse_intra_modality(
    classical_bundle: PredictionBundle,
    dl_bundle: PredictionBundle,
    modality_name: str,
    min_weight: float = 0.05,
) -> PredictionBundle:
    """Stage 1: fuse classical + DL for a single modality."""
    return _fuse(
        [classical_bundle, dl_bundle],
        fused_modality=modality_name,
        min_weight=min_weight,
        propagate_internal_sigma=False,
    )


def fuse_modalities(
    *modality_bundles: PredictionBundle,
    min_weight: float = 0.05,
) -> PredictionBundle:
    """Stage 2: fuse modality-level bundles into a final prediction."""
    return _fuse(
        list(modality_bundles),
        fused_modality="final_fusion",
        min_weight=min_weight,
        propagate_internal_sigma=True,
    )


def load_bundle_from_csv(
    csv_path: str | Path,
    modality: str,
    validation_mae: float,
) -> PredictionBundle:
    """Build a PredictionBundle from an inference CSV.

    Expected columns: ``record_name``, ``y_pred``.
    Optional: ``depth_mm`` (y_true), ``sigma``.
    """
    df = pd.read_csv(csv_path)
    sigma = df["sigma"].to_numpy() if "sigma" in df.columns else np.zeros(len(df))
    y_true = df["depth_mm"].to_numpy() if "depth_mm" in df.columns else None
    return PredictionBundle(
        modality=modality,
        record_names=df["record_name"].to_numpy(),
        y_pred=df["y_pred"].to_numpy(),
        sigma=sigma,
        validation_mae=validation_mae,
        y_true=y_true,
    )


def save_fusion_report(
    bundle: PredictionBundle,
    out_dir: str | Path,
) -> None:
    """Save predictions CSV + report/setup JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle.to_dataframe().to_csv(out_dir / "fusion_predictions.csv", index=False)

    metrics = bundle_batch_metrics(bundle)
    report: dict[str, Any] = {
        "modality": bundle.modality,
        "n_predictions": len(bundle.y_pred),
        "batch_mae_mm": metrics["mae_mm"],
        "batch_rmse_mm": metrics["rmse_mm"],
        "n_with_ground_truth": metrics["n_with_ground_truth"],
        "metric_definition": {
            "batch_mae_mm": "Mean absolute error on the current batch (mm).",
            "batch_rmse_mm": "Root mean squared error on the current batch (mm).",
        },
        "prediction_summary": {
            "mean_prediction_mm": float(np.mean(bundle.y_pred)) if len(bundle.y_pred) else None,
            "std_prediction_mm": float(np.std(bundle.y_pred)) if len(bundle.y_pred) else None,
            "mean_sigma_mm": float(np.mean(bundle.sigma)) if len(bundle.sigma) else None,
        },
    }
    if "source_batch_metrics" in bundle.metadata:
        report["source_batch_metrics"] = bundle.metadata["source_batch_metrics"]
    with open(out_dir / "fusion_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    meta = dict(bundle.metadata)
    source_modalities = list(meta.pop("source_modalities", []))
    source_maes = list(meta.pop("source_maes", []))
    weights = list(meta.pop("weights", []))
    meta.pop("source_batch_metrics", None)

    setup = {
        "modality": bundle.modality,
        "reference_mae_for_weighting_mm": float(bundle.validation_mae),
        "weighting_method": "inverse_reference_mae_with_floor",
        "source_modalities": source_modalities,
        "source_reference_maes_mm": source_maes,
        "weights": weights,
    }
    if meta:
        setup["metadata"] = meta
    with open(out_dir / "fusion_setup.json", "w", encoding="utf-8") as fh:
        json.dump(setup, fh, indent=2)
