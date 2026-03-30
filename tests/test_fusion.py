"""Tests for vm_micro.fusion  PredictionBundle, fuse_intra_modality, fuse_modalities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vm_micro.fusion.fuser import (
    PredictionBundle,
    fuse_intra_modality,
    fuse_modalities,
    hierarchical_sigma,
    load_bundle_from_csv,
    normalize_weights,
    save_fusion_report,
    weighted_sigma,
)

#
# Fixtures
#

_RECORDS = np.array([f"run01__seg{i:03d}" for i in range(20)])
_DEPTHS = np.linspace(0.1, 1.0, 20)
RNG = np.random.default_rng(7)


def _make_bundle(
    modality: str,
    val_mae: float,
    noise: float = 0.05,
    records: np.ndarray = _RECORDS,
    depths: np.ndarray = _DEPTHS,
) -> PredictionBundle:
    preds = depths + RNG.normal(0, noise, size=len(depths))
    sigma = np.full(len(depths), noise)
    return PredictionBundle(
        modality=modality,
        record_names=records,
        y_pred=preds,
        sigma=sigma,
        validation_mae=val_mae,
        y_true=depths.copy(),
    )


#
# PredictionBundle
#


def test_bundle_shapes():
    b = _make_bundle("test", 0.05)
    assert b.y_pred.shape == b.sigma.shape == b.record_names.shape
    assert b.y_true is not None


def test_bundle_to_dataframe():
    b = _make_bundle("test", 0.05)
    df = b.to_dataframe()
    assert "record_name" in df.columns
    assert "y_pred" in df.columns
    assert "sigma" in df.columns
    assert len(df) == len(_RECORDS)


#
# fuse_intra_modality
#


def test_intra_fusion_output_shape():
    cls_b = _make_bundle("airborne_classical", val_mae=0.060)
    dl_b = _make_bundle("airborne_dl", val_mae=0.045)
    fused = fuse_intra_modality(cls_b, dl_b, "airborne_ensemble")

    assert fused.modality == "airborne_ensemble"
    assert len(fused.y_pred) == len(_RECORDS)
    assert len(fused.sigma) == len(_RECORDS)


def test_intra_fusion_better_model_gets_higher_weight():
    """The DL model has lower MAE  higher weight  fused prediction closer to DL."""
    cls_b = _make_bundle("cls", val_mae=0.200, noise=0.0)
    dl_b = _make_bundle("dl", val_mae=0.050, noise=0.0)

    fused = fuse_intra_modality(cls_b, dl_b, "ensemble")
    w_cls = 1 / 0.200
    w_dl = 1 / 0.050
    w_dl_norm = w_dl / (w_cls + w_dl)
    # The fused prediction should be a weighted average
    expected = w_dl_norm * dl_b.y_pred + (1 - w_dl_norm) * cls_b.y_pred
    np.testing.assert_allclose(fused.y_pred, expected, rtol=1e-5)


def test_intra_fusion_sigma_propagation():
    cls_b = _make_bundle("cls", val_mae=0.060)
    dl_b = _make_bundle("dl", val_mae=0.040)
    fused = fuse_intra_modality(cls_b, dl_b, "ensemble")
    # _fused = sqrt((w1*1) + (w2*2))  must be positive
    assert np.all(fused.sigma >= 0)
    assert np.all(np.isfinite(fused.sigma))


def test_weighted_sigma_matches_definition():
    values = np.array([1.0, 3.0], dtype=np.float64)
    weights = np.array([0.75, 0.25], dtype=np.float64)
    mu = float(np.sum(weights * values))
    expected = float(np.sqrt(np.sum(weights * (values - mu) ** 2)))
    assert np.isclose(weighted_sigma(values, weights), expected, atol=1e-12)


def test_hierarchical_sigma_matches_definition():
    means = np.array([1.0, 3.0], dtype=np.float64)
    sigmas = np.array([0.2, 0.4], dtype=np.float64)
    weights = np.array([0.75, 0.25], dtype=np.float64)
    mu = float(np.sum(weights * means))
    expected = float(np.sqrt(np.sum(weights * (((means - mu) ** 2) + (sigmas**2)))))
    assert np.isclose(hierarchical_sigma(means, sigmas, weights), expected, atol=1e-12)


def test_normalize_weights_falls_back_to_equal_on_invalid():
    out = normalize_weights(np.array([np.nan, 3.0], dtype=np.float64))
    np.testing.assert_allclose(out, np.array([0.5, 0.5], dtype=np.float64), atol=1e-12)
    out2 = normalize_weights(np.array([0.0, 3.0], dtype=np.float64))
    np.testing.assert_allclose(out2, np.array([0.5, 0.5], dtype=np.float64), atol=1e-12)


#
# fuse_modalities (inter)
#


def test_inter_fusion_two_modalities():
    air = _make_bundle("airborne_ensemble", val_mae=0.040)
    struc = _make_bundle("structure_ensemble", val_mae=0.060)
    final = fuse_modalities(air, struc)

    assert final.modality == "final_fusion"
    assert len(final.y_pred) == len(_RECORDS)
    assert "weights" in final.metadata
    assert len(final.metadata["weights"]) == 2


def test_single_modality_passthrough():
    b = _make_bundle("airborne_ensemble", val_mae=0.040)
    final = fuse_modalities(b)
    np.testing.assert_array_equal(final.y_pred, b.y_pred)


def test_intra_fusion_sigma_is_weighted_disagreement():
    records = np.array(["r1", "r2"])
    cls = PredictionBundle(
        modality="airborne_classical",
        record_names=records,
        y_pred=np.array([1.0, 2.0]),
        sigma=np.array([0.0, 0.0]),
        validation_mae=0.1,
    )
    dl = PredictionBundle(
        modality="airborne_dl",
        record_names=records,
        y_pred=np.array([3.0, 4.0]),
        sigma=np.array([0.0, 0.0]),
        validation_mae=0.2,
    )
    fused = fuse_intra_modality(cls, dl, "airborne_ensemble")

    w = np.array([2.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    expected_mu = w[0] * cls.y_pred + w[1] * dl.y_pred
    expected_sigma = np.sqrt(
        w[0] * (cls.y_pred - expected_mu) ** 2 + w[1] * (dl.y_pred - expected_mu) ** 2
    )
    np.testing.assert_allclose(fused.y_pred, expected_mu, atol=1e-12)
    np.testing.assert_allclose(fused.sigma, expected_sigma, atol=1e-12)


def test_final_sigma_propagates_modality_disagreement_and_internal_sigma():
    records = np.array(["r1", "r2"])
    air = PredictionBundle(
        modality="airborne_ensemble",
        record_names=records,
        y_pred=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.2]),
        validation_mae=0.1,
    )
    st = PredictionBundle(
        modality="structure_ensemble",
        record_names=records,
        y_pred=np.array([3.0, 0.0]),
        sigma=np.array([0.3, 0.4]),
        validation_mae=0.2,
    )
    final = fuse_modalities(air, st)

    w = np.array([2.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    expected_mu = w[0] * air.y_pred + w[1] * st.y_pred
    expected_sigma = np.sqrt(
        w[0] * (((air.y_pred - expected_mu) ** 2) + air.sigma**2)
        + w[1] * (((st.y_pred - expected_mu) ** 2) + st.sigma**2)
    )
    np.testing.assert_allclose(final.y_pred, expected_mu, atol=1e-12)
    np.testing.assert_allclose(final.sigma, expected_sigma, atol=1e-12)


def test_missing_member_predictions_are_ignored_rowwise():
    records = np.array(["r1", "r2"])
    cls = PredictionBundle(
        modality="airborne_classical",
        record_names=records,
        y_pred=np.array([1.0, np.nan]),
        sigma=np.array([0.0, 0.0]),
        validation_mae=0.1,
    )
    dl = PredictionBundle(
        modality="airborne_dl",
        record_names=records,
        y_pred=np.array([3.0, 5.0]),
        sigma=np.array([0.0, 0.0]),
        validation_mae=0.2,
    )
    fused = fuse_intra_modality(cls, dl, "airborne_ensemble")
    np.testing.assert_allclose(fused.y_pred, np.array([5.0 / 3.0, 5.0]), atol=1e-12)
    np.testing.assert_allclose(fused.sigma, np.array([2.0 * np.sqrt(2.0) / 3.0, 0.0]), atol=1e-12)


#
# Record alignment
#


def test_fusion_aligns_records_to_intersection():
    records_a = np.array([f"seg{i:03d}" for i in range(10)])
    records_b = np.array([f"seg{i:03d}" for i in range(5, 15)])

    b_a = PredictionBundle("A", records_a, np.ones(10), np.ones(10) * 0.05, 0.05)
    b_b = PredictionBundle("B", records_b, np.ones(10), np.ones(10) * 0.05, 0.05)

    fused = fuse_intra_modality(b_a, b_b, "ensemble")
    # intersection = seg005..seg009  5 records
    assert len(fused.y_pred) == 5


#
# Persistence
#


def test_save_and_reload_report(tmp_path: Path):
    b = _make_bundle("airborne_ensemble", val_mae=0.040)
    save_fusion_report(b, tmp_path)

    assert (tmp_path / "fusion_predictions.csv").exists()
    assert (tmp_path / "fusion_report.json").exists()
    assert (tmp_path / "fusion_setup.json").exists()

    with open(tmp_path / "fusion_report.json") as fh:
        report = json.load(fh)
    assert report["modality"] == "airborne_ensemble"
    assert report["n_predictions"] == len(_RECORDS)
    assert report["n_with_ground_truth"] == len(_RECORDS)
    assert report["batch_mae_mm"] is not None  # y_true was provided
    assert report["batch_rmse_mm"] is not None

    with open(tmp_path / "fusion_setup.json") as fh:
        setup = json.load(fh)
    assert setup["modality"] == "airborne_ensemble"
    assert setup["reference_mae_for_weighting_mm"] == 0.04


def test_fused_report_includes_source_batch_metrics(tmp_path: Path):
    cls_b = _make_bundle("airborne_classical", val_mae=0.060)
    dl_b = _make_bundle("airborne_dl", val_mae=0.045)
    fused = fuse_intra_modality(cls_b, dl_b, "airborne_ensemble")
    save_fusion_report(fused, tmp_path)

    with open(tmp_path / "fusion_report.json", encoding="utf-8") as fh:
        report = json.load(fh)
    assert "source_batch_metrics" in report
    assert len(report["source_batch_metrics"]) == 2
    assert {item["modality"] for item in report["source_batch_metrics"]} == {
        "airborne_classical",
        "airborne_dl",
    }
    for item in report["source_batch_metrics"]:
        assert "mae_mm" in item
        assert "rmse_mm" in item

    with open(tmp_path / "fusion_setup.json", encoding="utf-8") as fh:
        setup = json.load(fh)
    assert setup["weighting_method"] == "inverse_reference_mae_with_floor"
    assert len(setup["weights"]) == 2
    assert len(setup["source_reference_maes_mm"]) == 2


def test_load_bundle_from_csv(tmp_path: Path):
    b = _make_bundle("airborne_ensemble", val_mae=0.040)
    save_fusion_report(b, tmp_path)

    reloaded = load_bundle_from_csv(
        tmp_path / "fusion_predictions.csv",
        modality="airborne_ensemble",
        validation_mae=0.040,
    )
    np.testing.assert_allclose(reloaded.y_pred, b.y_pred, rtol=1e-5)
