from __future__ import annotations

import asyncio
import concurrent.futures
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from shiny import App, reactive, render, ui

from app.db import DashboardDB
from app.parser import RunParser
from app.runner import FinalPredictionRunner
from app.settings import load_settings
from app.visualizations import (
    _message_figure,
    make_abs_residual_rank_figure,
    make_batch_model_quality_figure,
    make_batch_timeline_figure,
    make_confidence_mix_figure,
    make_hole_heatmap_figure,
    make_residual_scatter_figure,
    make_sigma_rank_figure,
    make_single_model_spread_figure,
    make_single_prediction_focus_figure,
    make_single_sources_figure,
    make_single_uncertainty_breakdown_figure,
    make_spectrogram_figure,
    make_uncertainty_scatter_figure,
    modality_prediction_rows_to_df,
    predictions_rows_to_df,
)

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

settings = load_settings()
db = DashboardDB(settings.db_path)
db.init()

pool = concurrent.futures.ThreadPoolExecutor(max_workers=6)


# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------


def _parse_actual_depth_mm(raw: str) -> float | None:
    text = str(raw).strip()
    if not text:
        return None
    return float(text)



def _normalize_scalar(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value

    return value



def _rows_to_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)



def _metrics_dict_to_df(metrics: dict[str, Any] | None) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for name, payload in metrics.items():
        row: dict[str, Any] = {"name": name}
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (dict, list)):
                    continue
                row[str(key)] = _normalize_scalar(value)
        else:
            row["value"] = _normalize_scalar(payload)
        rows.append(row)

    return pd.DataFrame(rows)



def _round_display_df(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


MODE_DE = {
    "single": "Einzel",
    "batch": "Batch",
    "none": "Keine",
}

STATUS_DE = {
    "ready": "Bereit",
    "running": "Läuft",
    "succeeded": "Erfolgreich",
    "failed": "Fehlgeschlagen",
    "pending": "Ausstehend",
    "queued": "In Warteschlange",
    "created": "Erstellt",
    "started": "Gestartet",
}

MODALITY_DE = {
    "airborne": "Luftschall",
    "structure": "Körperschall",
}

SOURCE_DE = {
    "classical": "Klassisch",
    "dl": "DL",
    "fusion": "Fusion",
}

CONFIDENCE_DE = {
    "high": "Hoch",
    "medium": "Mittel",
    "low": "Niedrig",
}

COLUMN_DE = {
    "id": "Lauf-ID",
    "name": "Name",
    "mode": "Modus",
    "status": "Status",
    "file_name": "Dateiname",
    "file_path": "Pfad",
    "detected_at_utc_plus_2": "Erkannt am",
    "created_at_utc_plus_2": "Erstellt am",
    "started_at_utc_plus_2": "Gestartet am",
    "finished_at_utc_plus_2": "Beendet am",
    "finished_at": "Beendet am",
    "airborne_file_name": "Luftschall-Datei",
    "structure_file_name": "Körperschall-Datei",
    "output_dir": "Ausgabeordner",
    "requested_output_dir": "Angeforderter Ausgabeordner",
    "resolved_run_dir": "Aufgelöster Laufordner",
    "final_dir": "Finaler Ordner",
    "record_name": "Datensatz",
    "step_idx": "Schritt",
    "hole_id": "Bohrung",
    "y_pred": "Prognose [mm]",
    "y_pred_mm": "Prognose [mm]",
    "rounded_step_mm": "Gerundete Stufe [mm]",
    "sigma": "Sigma [mm]",
    "sigma_mm": "Sigma [mm]",
    "confidence_label": "Konfidenz",
    "y_true": "Ist-Tiefe [mm]",
    "actual_depth_mm": "Ist-Tiefe [mm]",
    "residual_mm": "Residuum [mm]",
    "abs_residual_mm": "Absolutes Residuum [mm]",
    "z_abs": "|z|",
    "within_1sigma": "Innerhalb 1σ",
    "within_2sigma": "Innerhalb 2σ",
    "sigma_airborne_mm": "Sigma Luftschall [mm]",
    "sigma_structure_mm": "Sigma Körperschall [mm]",
    "sigma_between_modalities_mm": "Sigma zwischen Modalitäten [mm]",
    "available_modalities": "Verfügbare Modalitäten",
    "n_predictions": "Anzahl Prognosen",
    "n_with_ground_truth": "Anzahl mit Ist-Wert",
    "mae_mm": "MAE [mm]",
    "rmse_mm": "RMSE [mm]",
    "sigma_mean_mm": "Mittleres Sigma [mm]",
    "sigma_median_mm": "Median Sigma [mm]",
    "sigma_max_mm": "Maximales Sigma [mm]",
    "low_conf_count": "Anzahl niedriger Konfidenz",
    "coverage_1sigma": "Abdeckung 1σ",
    "coverage_2sigma": "Abdeckung 2σ",
    "modality": "Modalität",
    "source": "Quelle",
    "spectrogram_source": "Spektrogrammquelle",
    "raw_source_file": "Rohdatei",
    "features_csv": "Merkmals-CSV",
    "classical_predictions_csv": "Klassische Prognosen CSV",
    "dl_predictions_csv": "DL-Prognosen CSV",
    "fusion_predictions_csv": "Fusions-Prognosen CSV",
    "debug_core_count": "Anzahl Debug Core",
    "debug_padded_count": "Anzahl Debug Padded",
    "run_id": "Lauf-ID",
    "value": "Wert",
    "validation_mae": "Validierungs-MAE",
    "setup_audit_json": "Setup-Audit JSON",
    "final_predictions_csv": "Finale Prognosen CSV",
}


def _translate_mode(value: Any) -> Any:
    key = str(value).strip().lower()
    return MODE_DE.get(key, value)



def _translate_status(value: Any) -> Any:
    key = str(value).strip().lower()
    return STATUS_DE.get(key, value)



def _translate_modality(value: Any) -> Any:
    key = str(value).strip().lower()
    return MODALITY_DE.get(key, value)



def _translate_source(value: Any) -> Any:
    key = str(value).strip().lower()
    return SOURCE_DE.get(key, value)



def _translate_confidence(value: Any) -> Any:
    key = str(value).strip().lower()
    return CONFIDENCE_DE.get(key, value)



def _translate_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return "Ja" if value else "Nein"
    return value



def _translate_modalities_string(value: Any) -> Any:
    text = str(value).strip()
    if not text:
        return value
    parts = [part.strip() for part in text.split(',') if part.strip()]
    if not parts:
        return value
    return ', '.join(str(_translate_modality(part)) for part in parts)



def _translate_display_value(column: str, value: Any) -> Any:
    if value is None:
        return value
    if pd.isna(value):
        return value
    if column in {"mode"}:
        return _translate_mode(value)
    if column in {"status"}:
        return _translate_status(value)
    if column in {"modality"}:
        return _translate_modality(value)
    if column in {"source"}:
        return _translate_source(value)
    if column in {"confidence_label"}:
        return _translate_confidence(value)
    if column in {"available_modalities"}:
        return _translate_modalities_string(value)
    if column.startswith("within_"):
        return _translate_bool(value)
    if isinstance(value, bool):
        return _translate_bool(value)
    return value



def _localize_display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda v, c=col: _translate_display_value(c, v))
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(_translate_bool)
    out = out.rename(columns={col: COLUMN_DE.get(col, col) for col in out.columns})
    return out



def _fmt_detected_file(row: dict[str, Any] | None, label: str) -> str:
    label_de = str(_translate_modality(label))
    if row is None:
        return f"{label_de}: keine"
    lines = [
        f"{label_de}: {row.get('file_name')}",
        f"Status: {_translate_status(row.get('status'))}",
        f"Erkannt am: {row.get('detected_at_utc_plus_2')}",
        f"Pfad: {row.get('file_path')}",
    ]
    return "\n".join(lines)



def _fmt_active_run(row: dict[str, Any] | None) -> str:
    if row is None:
        return "Kein aktiver Lauf."
    lines = [
        f"Lauf-ID: {row.get('id')}",
        f"Modus: {_translate_mode(row.get('mode'))}",
        f"Status: {_translate_status(row.get('status'))}",
        f"Erstellt am: {row.get('created_at_utc_plus_2')}",
        f"Gestartet am: {row.get('started_at_utc_plus_2')}",
        f"Luftschall: {row.get('airborne_file_name')}",
        f"Körperschall: {row.get('structure_file_name')}",
    ]
    return "\n".join(lines)



def _fmt_latest_success(row: dict[str, Any] | None) -> str:
    if row is None:
        return "Noch kein abgeschlossener Lauf."
    lines = [
        f"Lauf-ID: {row.get('id')}",
        f"Modus: {_translate_mode(row.get('mode'))}",
        f"Status: {_translate_status(row.get('status'))}",
        f"Beendet am: {row.get('finished_at_utc_plus_2')}",
        f"Ausgabeordner: {row.get('output_dir')}",
    ]
    return "\n".join(lines)



def _read_history_rows() -> list[dict[str, Any]]:
    parser_db = DashboardDB(settings.db_path)
    parser_db.init()
    parser = RunParser(settings=settings, db=parser_db)
    return parser.list_history(limit=settings.history_limit)



def _history_run_label(row: dict[str, Any]) -> str:
    run_id = row.get("id")
    finished_at = row.get("finished_at") or "-"
    mode = _translate_mode(row.get("mode") or "-")
    status = _translate_status(row.get("status") or "-")
    airborne = row.get("airborne_file_name") or "-"
    return f"{run_id} | {finished_at} | {mode} | {status} | {airborne}"



def _history_run_choices(rows: list[dict[str, Any]]) -> dict[str, str]:
    choices: dict[str, str] = {}
    for row in rows:
        if row.get("status") != "succeeded":
            continue
        run_id = row.get("id")
        if run_id is None:
            continue
        choices[str(run_id)] = _history_run_label(row)
    return choices



def _current_display_text(selected_history_run_id: int | None, latest_success: dict[str, Any] | None) -> str:
    if selected_history_run_id is not None:
        return f"Angezeigt wird historischer Lauf mit Lauf-ID={selected_history_run_id}"
    if latest_success is None:
        return "Angezeigt wird der neueste erfolgreiche Lauf: keiner verfügbar"
    return f"Angezeigt wird der neueste erfolgreiche Lauf mit Lauf-ID={latest_success.get('id')}"



def _latest_succeeded_run(db_obj: DashboardDB) -> dict[str, Any] | None:
    for row in db_obj.list_runs(limit=settings.history_limit):
        if row["status"] == "succeeded":
            return row
    return None



def _read_shell_snapshot() -> dict[str, Any]:
    latest_files = db.get_latest_detected_files()
    airborne = latest_files.get("airborne")
    structure = latest_files.get("structure")
    active_run = db.get_active_run()
    latest_success = _latest_succeeded_run(db)

    ready = (
        active_run is None
        and airborne is not None
        and structure is not None
        and airborne.get("status") == "ready"
        and structure.get("status") == "ready"
    )

    return {
        "airborne": airborne,
        "structure": structure,
        "active_run": active_run,
        "latest_success": latest_success,
        "ready": ready,
    }



def _run_latest_detected(mode: str, actual_depth_mm: float | None) -> dict[str, Any]:
    runner_db = DashboardDB(settings.db_path)
    runner_db.init()
    runner = FinalPredictionRunner(settings=settings, db=runner_db)
    result = runner.run_latest_detected(mode=mode, actual_depth_mm=actual_depth_mm)

    parser_db = DashboardDB(settings.db_path)
    parser_db.init()
    parser = RunParser(settings=settings, db=parser_db)
    parsed = parser.parse_run(result.run_id)

    return {
        "run_id": result.run_id,
        "mode": mode,
        "output_dir": str(result.output_dir) if result.output_dir else None,
        "resolved_run_dir": parsed["run"]["resolved_run_dir"],
    }



def _existing_path_or_none(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve(strict=False)
    return str(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Dashboard-specific dataframe helpers
# ---------------------------------------------------------------------------


def _dashboard_mode(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "none"
    return str(payload.get("run", {}).get("mode") or "none")



def _display_predictions_df(payload: dict[str, Any] | None) -> pd.DataFrame:
    return predictions_rows_to_df((payload or {}).get("final_predictions"))



def _display_predictions_long_df(payload: dict[str, Any] | None) -> pd.DataFrame:
    return _round_display_df(_rows_to_df((payload or {}).get("final_predictions_long")))



def _segment_choices(payload: dict[str, Any] | None) -> list[str]:
    df = _display_predictions_df(payload)
    if df.empty or "record_name" not in df.columns:
        return []
    return [str(v) for v in df["record_name"].tolist()]



def _effective_selected_key(payload: dict[str, Any] | None, requested_key: str | None) -> str | None:
    choices = _segment_choices(payload)
    if not choices:
        return None
    if requested_key in choices:
        return requested_key
    return choices[0]



def _selected_prediction_df(payload: dict[str, Any] | None, selected_key: str | None) -> pd.DataFrame:
    df = _display_predictions_df(payload)
    if df.empty:
        return df
    mode = _dashboard_mode(payload)
    if mode == "single":
        return df.head(1)
    key = _effective_selected_key(payload, selected_key)
    if key is None:
        return df.head(1)
    match = df[df["record_name"].astype(str) == str(key)].copy()
    return match if not match.empty else df.head(1)



def _single_source_df(payload: dict[str, Any] | None, selected_key: str | None) -> pd.DataFrame:
    return modality_prediction_rows_to_df(payload, selected_key)



def _summary_metrics_df(payload: dict[str, Any] | None) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()

    summary = payload["summary"]
    run = payload["run"]
    mode = _dashboard_mode(payload)
    pred_df = _display_predictions_df(payload)

    if mode == "single":
        row = pred_df.iloc[0] if not pred_df.empty else pd.Series(dtype=object)
        return _round_display_df(
            pd.DataFrame(
                [
                    {
                        "mode": summary.get("mode"),
                        "actual_depth_mm": summary.get("actual_depth_mm"),
                        "available_modalities": ", ".join(run.get("available_modalities", [])),
                        "y_pred_mm": row.get("y_pred"),
                        "rounded_step_mm": row.get("rounded_step_mm"),
                        "sigma_mm": row.get("sigma"),
                        "confidence_label": row.get("confidence_label"),
                        "residual_mm": row.get("residual_mm"),
                    }
                ]
            )
        )

    final_fusion = summary.get("final_fusion", {}) or {}
    low_conf_count = int(pred_df["confidence_label"].astype(str).eq("low").sum()) if (not pred_df.empty and "confidence_label" in pred_df.columns) else None
    sigma_mean = float(pred_df["sigma"].mean()) if (not pred_df.empty and "sigma" in pred_df.columns) else None
    sigma_median = float(pred_df["sigma"].median()) if (not pred_df.empty and "sigma" in pred_df.columns) else None
    sigma_max = float(pred_df["sigma"].max()) if (not pred_df.empty and "sigma" in pred_df.columns) else None
    return _round_display_df(
        pd.DataFrame(
            [
                {
                    "mode": summary.get("mode"),
                    "available_modalities": ", ".join(run.get("available_modalities", [])),
                    "n_predictions": len(pred_df),
                    "n_with_ground_truth": final_fusion.get("n_with_ground_truth"),
                    "mae_mm": final_fusion.get("mae_mm"),
                    "rmse_mm": final_fusion.get("rmse_mm"),
                    "sigma_mean_mm": sigma_mean,
                    "sigma_median_mm": sigma_median,
                    "sigma_max_mm": sigma_max,
                    "low_conf_count": low_conf_count,
                    "coverage_1sigma": final_fusion.get("coverage_1sigma"),
                    "coverage_2sigma": final_fusion.get("coverage_2sigma"),
                }
            ]
        )
    )



def _prediction_focus_df(payload: dict[str, Any] | None, selected_key: str | None) -> pd.DataFrame:
    df = _selected_prediction_df(payload, selected_key)
    wanted = [
        c
        for c in (
            "record_name",
            "step_idx",
            "hole_id",
            "y_pred",
            "rounded_step_mm",
            "sigma",
            "confidence_label",
            "y_true",
            "residual_mm",
            "abs_residual_mm",
            "z_abs",
            "within_1sigma",
            "within_2sigma",
            "sigma_airborne_mm",
            "sigma_structure_mm",
            "sigma_between_modalities_mm",
        )
        if c in df.columns
    ]
    return _round_display_df(df[wanted] if not df.empty else pd.DataFrame())



def _single_uncertainty_df(payload: dict[str, Any] | None, selected_key: str | None) -> pd.DataFrame:
    df = _selected_prediction_df(payload, selected_key)
    if df.empty:
        return pd.DataFrame()
    wanted = [
        c
        for c in (
            "sigma",
            "sigma_airborne_mm",
            "sigma_structure_mm",
            "sigma_between_modalities_mm",
            "z_abs",
            "within_1sigma",
            "within_2sigma",
            "confidence_label",
        )
        if c in df.columns
    ]
    return _round_display_df(df[wanted])



def _top_uncertainty_df(payload: dict[str, Any] | None) -> pd.DataFrame:
    df = _display_predictions_df(payload)
    if df.empty or "sigma" not in df.columns:
        return pd.DataFrame()
    wanted = [
        c
        for c in (
            "record_name",
            "step_idx",
            "hole_id",
            "y_pred",
            "sigma",
            "confidence_label",
            "residual_mm",
            "abs_residual_mm",
        )
        if c in df.columns
    ]
    return _round_display_df(df.sort_values("sigma", ascending=False)[wanted].head(15))



def _artifact_focus_df(payload: dict[str, Any] | None, selected_key: str | None) -> pd.DataFrame:
    run = (payload or {}).get("run", {})
    focus = _prediction_focus_df(payload, selected_key)
    if focus.empty:
        return pd.DataFrame()
    extra = {
        "mode": run.get("mode"),
        "available_modalities": ", ".join(run.get("available_modalities", [])),
    }
    for key, value in extra.items():
        focus[key] = value
    return _round_display_df(focus)



def _resolve_spectrogram_source(payload: dict[str, Any] | None, modality: str, selected_key: str | None) -> str | None:
    if payload is None:
        return None

    mode = payload["run"]["mode"]
    mod_payload = payload["modalities"].get(modality, {})
    if not mod_payload.get("present"):
        return None

    if mode == "single":
        return _existing_path_or_none(mod_payload.get("raw_source_file_path"))

    segment_map = mod_payload.get("segment_file_map", {}) or {}
    if selected_key and selected_key in segment_map:
        return _existing_path_or_none(segment_map[selected_key])

    if segment_map:
        first_key = sorted(segment_map.keys())[0]
        return _existing_path_or_none(segment_map[first_key])

    return None



def _artifact_paths_df(payload: dict[str, Any] | None, selected_key: str | None) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for modality in ("airborne", "structure"):
        mod_payload = payload["modalities"].get(modality, {})
        rows.append(
            {
                "modality": modality,
                "spectrogram_source": _resolve_spectrogram_source(payload, modality, selected_key),
                "raw_source_file": mod_payload.get("raw_source_file_path"),
                "features_csv": mod_payload.get("features_csv_path"),
                "classical_predictions_csv": mod_payload.get("classical_predictions_csv"),
                "dl_predictions_csv": mod_payload.get("dl_predictions_csv"),
                "fusion_predictions_csv": mod_payload.get("fusion_predictions_csv"),
                "debug_core_count": len(mod_payload.get("debug_core_paths", [])),
                "debug_padded_count": len(mod_payload.get("debug_padded_paths", [])),
            }
        )
    return pd.DataFrame(rows)

def _batch_model_quality_df(payload: dict[str, Any] | None) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()

    summary = (payload or {}).get("summary", {}) or {}
    rows: list[dict[str, Any]] = []

    models = summary.get("models", {}) or {}
    model_map = {
        "airborne_classical": ("Luftschall · Klassisch", "classical"),
        "airborne_dl": ("Luftschall · DL", "dl"),
        "structure_classical": ("Körperschall · Klassisch", "classical"),
        "structure_dl": ("Körperschall · DL", "dl"),
    }

    for name, metrics in models.items():
        if not isinstance(metrics, dict):
            continue

        label, color_key = model_map.get(
            str(name).strip().lower(),
            (str(name), "muted"),
        )

        rows.append(
            {
                "label": label,
                "mae_mm": pd.to_numeric(metrics.get("mae_mm"), errors="coerce"),
                "spread_mm": pd.to_numeric(metrics.get("rmse_mm"), errors="coerce"),
                "color_key": color_key,
            }
        )

    modality_fusions = summary.get("modality_fusions", {}) or {}
    fusion_map = {
        "airborne_ensemble": ("Luftschall · Fusion", "airborne"),
        "structure_ensemble": ("Körperschall · Fusion", "structure"),
    }

    for name, metrics in modality_fusions.items():
        if not isinstance(metrics, dict):
            continue

        label, color_key = fusion_map.get(
            str(name).strip().lower(),
            (str(name), "muted"),
        )

        rows.append(
            {
                "label": label,
                "mae_mm": pd.to_numeric(metrics.get("mae_mm"), errors="coerce"),
                "spread_mm": pd.to_numeric(metrics.get("rmse_mm"), errors="coerce"),
                "color_key": color_key,
            }
        )

    final_fusion = summary.get("final_fusion", {}) or {}
    if isinstance(final_fusion, dict) and final_fusion:
        rows.append(
            {
                "label": "Finale Fusion",
                "mae_mm": pd.to_numeric(final_fusion.get("mae_mm"), errors="coerce"),
                "spread_mm": pd.to_numeric(final_fusion.get("rmse_mm"), errors="coerce"),
                "color_key": "final",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["mae_mm"] = pd.to_numeric(df["mae_mm"], errors="coerce")
    df["spread_mm"] = pd.to_numeric(df["spread_mm"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["mae_mm"]).sort_values("mae_mm", ascending=True).reset_index(drop=True)
    return df


def _run_metadata_text(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "Noch keine geparsten Laufmetadaten verfügbar."
    run = payload["run"]
    report_paths = payload["report_paths"]
    lines = [
        f"DB-Lauf-ID: {run.get('db_run_id')}",
        f"Modus: {_translate_mode(run.get('mode'))}",
        f"Status: {_translate_status(run.get('status'))}",
        f"Angeforderter Ausgabeordner: {run.get('requested_output_dir')}",
        f"Aufgelöster Laufordner: {run.get('resolved_run_dir')}",
        f"Finaler Ordner: {run.get('final_dir')}",
        f"Verfügbare Modalitäten: {_translate_modalities_string(', '.join(run.get('available_modalities', [])))}",
        f"Setup-Audit JSON: {report_paths.get('setup_audit_json')}",
        f"Finale Prognosen CSV: {report_paths.get('final_predictions_csv')}",
    ]
    return "\n".join(lines)



def _audit_warnings_text(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "Noch kein Audit verfügbar."
    warnings = payload.get("audit", {}).get("warnings", []) or []
    if not warnings:
        return "Keine Setup-Warnungen."
    return "\n".join(str(w) for w in warnings)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _kpi_card(title: str, value: str, subtitle: str) -> Any:
    return ui.card(
        ui.card_header(title),
        ui.tags.h3(value),
        ui.tags.p(subtitle),
    )



def _kpi_row_ui(payload: dict[str, Any] | None, selected_key: str | None) -> Any:
    if payload is None:
        return ui.card(ui.card_header("Zusammenfassung"), ui.p("Noch kein erfolgreicher Lauf verfügbar."))

    mode = _dashboard_mode(payload)
    pred_df = _display_predictions_df(payload)
    focus_df = _selected_prediction_df(payload, selected_key)

    if mode == "single":
        row = focus_df.iloc[0] if not focus_df.empty else pd.Series(dtype=object)
        cards = [
            _kpi_card("Finale Prognose", f"{row.get('y_pred', '—'):.4f} mm" if pd.notna(row.get("y_pred")) else "—", "Neueste fusionierte Prognose"),
            _kpi_card("Sigma", f"{row.get('sigma', '—'):.4f} mm" if pd.notna(row.get("sigma")) else "—", "Geschätzte Unsicherheit"),
            _kpi_card("Konfidenz", str(_translate_confidence(row.get("confidence_label", "—"))), "Hoch / Mittel / Niedrig")
        ]
        if "residual_mm" in row and pd.notna(row.get("residual_mm")):
            cards.append(_kpi_card("Residuum", f"{row.get('residual_mm'):.4f} mm", "Prognose minus Ist-Wert"))
        return ui.layout_columns(*cards, col_widths=(3,) * len(cards))

    low_conf_count = int(pred_df["confidence_label"].astype(str).eq("low").sum()) if (not pred_df.empty and "confidence_label" in pred_df.columns) else 0
    sigma_mean = float(pred_df["sigma"].mean()) if (not pred_df.empty and "sigma" in pred_df.columns) else None
    sigma_max = float(pred_df["sigma"].max()) if (not pred_df.empty and "sigma" in pred_df.columns) else None
    final_fusion = payload.get("summary", {}).get("final_fusion", {}) or {}
    coverage_1 = final_fusion.get("coverage_1sigma")
    mae = final_fusion.get("mae_mm")

    cards = [
        _kpi_card("Prognosen", str(len(pred_df)), "Zeilen im aktuellen fusionierten Batch"),
        _kpi_card("Niedrige Konfidenz", str(low_conf_count), "Zeilen mit der schwächsten Konfidenz"),
        _kpi_card("Mittleres Sigma", f"{sigma_mean:.4f} mm" if sigma_mean is not None else "—", "Durchschnittliche Unsicherheit im Batch"),
        _kpi_card("Höchstes Sigma", f"{sigma_max:.4f} mm" if sigma_max is not None else "—", "Zeile mit der höchsten Unsicherheit"),
        _kpi_card("1σ-Abdeckung", f"{coverage_1:.1%}" if coverage_1 is not None else "—", "Anteil der Residuen innerhalb eines Sigma"),
    ]
    if mae is not None:
        cards.append(_kpi_card("Batch-MAE", f"{float(mae):.4f} mm", "Fehler über Zeilen mit Ist-Wert"))
    return ui.layout_columns(*cards, col_widths=(2,) * len(cards))



def _grid(df: pd.DataFrame) -> Any:
    return render.DataGrid(_localize_display_df(_round_display_df(df)), width="100%")


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Laufsteuerung"),
        ui.input_text("actual_depth_mm", "Ist-Tiefe (optional, mm)", placeholder="z. B. 0.2"),
        ui.layout_columns(
            ui.input_task_button("run_single_btn", "Einzelrun starten"),
            ui.input_task_button("run_batch_btn", "Batchlauf starten"),
            col_widths=(6, 6),
        ),
        ui.input_select("artifact_key", "Artefakt- / Prognoseschlüssel", choices=[], selected=None),
        ui.hr(),
        ui.h5("Verlauf"),
        ui.input_select("history_run_id", "Erfolgreicher Lauf", choices={}, selected=None),
        ui.layout_columns(
            ui.input_action_button("load_history_run_btn", "Gewählten Lauf laden"),
            ui.input_action_button("follow_latest_btn", "Neuester Lauf"),
            col_widths=(6, 6),
        ),
        ui.output_text_verbatim("current_display_text", placeholder=True),
        ui.hr(),
        ui.h5("Bereitschaft"),
        ui.output_text_verbatim("readiness_text", placeholder=True),
        width=340,
    ),
    ui.h2("VM-Dashboard"),
    ui.navset_tab(
        ui.nav_panel(
            "Übersicht",
            ui.div(style="height: 0.6rem;"),
            ui.output_ui("kpi_row"),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Prognosefokus"),
                    ui.output_plot("prediction_focus_plot", height="320px"),
                    ui.output_data_frame("prediction_focus_table"),
                ),
                ui.card(
                    ui.card_header("Quellenvergleich"),
                    ui.output_plot("single_sources_plot", height="320px"),
                    full_screen=True
                ),
                col_widths=(7, 5),
            ),
            ui.output_ui("mode_specific_dashboard"),
        ),
        ui.nav_panel(
            "Artefakte",
            ui.div(style="height: 0.6rem;"),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Artefaktfokus"),
                    ui.output_data_frame("artifact_focus_table"),
                ),
                ui.card(
                    ui.card_header("Artefaktpfade"),
                    ui.output_data_frame("artifact_paths_table"),
                ),
                col_widths=(6, 6),
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Luftschall-Spektrogramm"),
                    ui.output_plot("airborne_spectrogram_plot", height="500px"),
                ),
                ui.card(
                    ui.card_header("Körperschall-Spektrogramm"),
                    ui.output_plot("structure_spectrogram_plot", height="500px"),
                ),
                col_widths=(6, 6),
            ),
            ui.card(
                ui.card_header("Debug-Plots"),
                ui.output_ui("debug_images_stack"),
                full_screen=True
            ),
            ui.card(
                ui.card_header("Laufmetadaten"),
                ui.output_text_verbatim("run_metadata_text", placeholder=True),
            ),
        ),
        ui.nav_panel(
            "Tabellen",
            ui.div(style="height: 0.6rem;"),
            ui.card(
                ui.card_header("Finale fusionierte Prognosen"),
                ui.output_data_frame("final_predictions_table"),
                full_screen=True
            ),
            ui.card(
                    ui.card_header("Zusammenfassende Kennzahlen"),
                    ui.output_data_frame("summary_metrics_table"),
                    full_screen=True
                ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Zeilen mit höchster Unsicherheit"),
                    ui.output_data_frame("top_uncertainty_table"),
                    full_screen=True
                ),
                ui.card(
                    ui.card_header("Modellkennzahlen"),
                    ui.output_data_frame("models_table"),
                    full_screen=True
                ),
                ui.card(
                    ui.card_header("Modalitäts-Fusionskennzahlen"),
                    ui.output_data_frame("modality_fusions_table"),
                    full_screen=True
                ),
                col_widths=(4, 4, 4),
            ),
            # ui.card(
            #     ui.card_header("Long-format prediction breakdown"),
            #     ui.output_data_frame("final_predictions_long_table"),
            #     full_screen=True
            # ),
            ui.card(
                ui.card_header("Letzte Läufe"),
                ui.output_data_frame("history_table"),
                full_screen=True
            ),
        ),
        ui.nav_panel(
            "Betrieb",
            ui.div(style="height: 0.6rem;"),
            ui.layout_columns(
                ui.card(ui.card_header("UI-Startstatus"), ui.output_text_verbatim("launch_result_text", placeholder=True)),
                ui.card(ui.card_header("Dashboard-Fehler"), ui.output_text_verbatim("dashboard_error_text", placeholder=True)),
                col_widths=(6, 6),
            ),
            ui.layout_columns(
                ui.card(ui.card_header("Neueste Luftschall-Datei"), ui.output_text_verbatim("latest_airborne_text", placeholder=True)),
                ui.card(ui.card_header("Neueste Körperschall-Datei"), ui.output_text_verbatim("latest_structure_text", placeholder=True)),
                col_widths=(6, 6),
            ),
            ui.layout_columns(
                ui.card(ui.card_header("Aktueller Lauf"), ui.output_text_verbatim("active_run_text", placeholder=True)),
                ui.card(ui.card_header("Letzter erfolgreicher Lauf"), ui.output_text_verbatim("latest_success_text", placeholder=True)),
                col_widths=(6, 6),
            ),
            ui.card(
                ui.card_header("Setup-Warnungen"),
                ui.output_text_verbatim("audit_warnings_text", placeholder=True),
            ),
        ),
    ),
    title="VM-Dashboard",
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def server(input, output, session):
    launch_note = reactive.value("In dieser Sitzung wurde kein Lauf über die UI gestartet.")
    parsed_payload_cache = reactive.value(None)
    parsed_payload_run_id = reactive.value(None)
    selected_history_run_id = reactive.value(None)
    history_choices_cache = reactive.value({})
    ui_error_message = reactive.value("")
    segment_choices_cache = reactive.value([])

    @reactive.calc
    def shell_snapshot() -> dict[str, Any]:
        reactive.invalidate_later(2)
        return _read_shell_snapshot()

    @reactive.calc
    def history_rows() -> list[dict[str, Any]]:
        shell_snapshot()
        return _read_history_rows()

    @reactive.effect
    def _refresh_history_choices():
        rows = history_rows()
        choices = _history_run_choices(rows)
        previous = history_choices_cache.get()
        if choices == previous:
            return
        current_selected = selected_history_run_id.get()
        selected_value = str(current_selected) if current_selected is not None and str(current_selected) in choices else None
        ui.update_select("history_run_id", choices=choices, selected=selected_value, session=session)
        history_choices_cache.set(dict(choices))

    @reactive.effect
    def _refresh_parsed_payload_when_run_changes():
        snap = shell_snapshot()
        latest_success = snap["latest_success"]
        selected_run_id = selected_history_run_id.get()

        if selected_run_id is not None:
            target_run_id: int | None = int(selected_run_id)
        elif latest_success is not None:
            target_run_id = int(latest_success["id"])
        else:
            target_run_id = None

        if target_run_id is None:
            parsed_payload_cache.set(None)
            parsed_payload_run_id.set(None)
            ui_error_message.set("")
            return

        if parsed_payload_run_id.get() == target_run_id:
            return

        parser_db = DashboardDB(settings.db_path)
        parser_db.init()
        parser = RunParser(settings=settings, db=parser_db)

        try:
            payload = parser.parse_run(target_run_id)
        except Exception as exc:
            parsed_payload_cache.set(None)
            parsed_payload_run_id.set(None)
            ui_error_message.set(f"Lauf {target_run_id} konnte nicht geparst werden: {exc}")
            return

        parsed_payload_cache.set(payload)
        parsed_payload_run_id.set(target_run_id)
        ui_error_message.set("")

        if launch_note.get().startswith(("Einzelrun", "Batchlauf")):
            launch_note.set(f"Lauf-ID={target_run_id} wurde ins Dashboard geladen.")

    @reactive.calc
    def latest_dashboard_payload() -> dict[str, Any] | None:
        return parsed_payload_cache.get()

    @reactive.effect
    def _refresh_artifact_choices_on_new_run():
        payload = latest_dashboard_payload()
        previous = segment_choices_cache.get()

        if payload is None:
            if previous:
                ui.update_select("artifact_key", choices=[], selected=None, session=session)
                segment_choices_cache.set([])
            return

        choices = _segment_choices(payload)
        if choices == previous:
            return
        selected = choices[0] if choices else None
        ui.update_select("artifact_key", choices=choices, selected=selected, session=session)
        segment_choices_cache.set(list(choices))

    @reactive.calc
    def effective_key() -> str | None:
        return _effective_selected_key(latest_dashboard_payload(), input.artifact_key())

    @ui.bind_task_button(button_id="run_single_btn")
    @reactive.extended_task
    async def run_single_task(actual_depth_mm: float | None):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(pool, _run_latest_detected, "single", actual_depth_mm)

    @ui.bind_task_button(button_id="run_batch_btn")
    @reactive.extended_task
    async def run_batch_task(actual_depth_mm: float | None):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(pool, _run_latest_detected, "batch", actual_depth_mm)

    @reactive.effect
    @reactive.event(input.run_single_btn)
    def _launch_single():
        try:
            actual_depth_mm = _parse_actual_depth_mm(input.actual_depth_mm())
            ui_error_message.set("")
            launch_note.set("Einzelrun wird gestartet...")
            run_single_task(actual_depth_mm)
        except Exception as exc:
            launch_note.set(f"Einzelrun wurde nicht gestartet: {exc}")

    @reactive.effect
    @reactive.event(input.run_batch_btn)
    def _launch_batch():
        try:
            actual_depth_mm = _parse_actual_depth_mm(input.actual_depth_mm())
            ui_error_message.set("")
            launch_note.set("Batchlauf wird gestartet...")
            run_batch_task(actual_depth_mm)
        except Exception as exc:
            launch_note.set(f"Batchlauf wurde nicht gestartet: {exc}")

    @reactive.effect
    @reactive.event(input.load_history_run_btn)
    def _load_selected_history_run():
        raw_value = input.history_run_id()
        if raw_value is None:
            return
        try:
            selected_history_run_id.set(int(str(raw_value).strip()))
        except ValueError:
            return

    @reactive.effect
    @reactive.event(input.follow_latest_btn)
    def _follow_latest():
        selected_history_run_id.set(None)

    # ---------------- Text outputs ----------------

    @render.text
    def readiness_text():
        snap = shell_snapshot()
        airborne = snap["airborne"]
        structure = snap["structure"]
        active_run = snap["active_run"]
        lines = [
            f"Startbereit: {'Ja' if snap['ready'] else 'Nein'}",
            f"Luftschall vorhanden: {'Ja' if airborne is not None else 'Nein'}",
            f"Körperschall vorhanden: {'Ja' if structure is not None else 'Nein'}",
            f"Aktiver Lauf: {'Ja' if active_run is not None else 'Nein'}",
        ]
        if airborne is not None:
            lines.append(f"Luftschall-Status: {_translate_status(airborne.get('status'))}")
        if structure is not None:
            lines.append(f"Körperschall-Status: {_translate_status(structure.get('status'))}")
        return "\n".join(lines)

    @render.text
    def latest_airborne_text():
        return _fmt_detected_file(shell_snapshot()["airborne"], "airborne")

    @render.text
    def latest_structure_text():
        return _fmt_detected_file(shell_snapshot()["structure"], "structure")

    @render.text
    def active_run_text():
        return _fmt_active_run(shell_snapshot()["active_run"])

    @render.text
    def latest_success_text():
        return _fmt_latest_success(shell_snapshot()["latest_success"])

    @render.text
    def dashboard_error_text():
        message = ui_error_message.get().strip()
        return message if message else "Keine Dashboard-Fehler."

    @render.text
    def launch_result_text():
        return launch_note.get()

    @render.text
    def current_display_text():
        snap = shell_snapshot()
        return _current_display_text(selected_history_run_id.get(), snap["latest_success"])

    @render.text
    def run_metadata_text():
        return _run_metadata_text(latest_dashboard_payload())

    @render.text
    def audit_warnings_text():
        return _audit_warnings_text(latest_dashboard_payload())

    # ---------------- UI outputs ----------------

    @render.ui
    def kpi_row():
        return _kpi_row_ui(latest_dashboard_payload(), effective_key())

    @render.ui
    def mode_specific_dashboard():
        mode = _dashboard_mode(latest_dashboard_payload())

        if mode == "single":
            return ui.tags.div(
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Unsicherheitszerlegung"),
                        ui.output_plot("single_uncertainty_breakdown_plot", height="320px"),
                        full_screen=True,
                    ),
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Modellstreuung und Einigung"),
                        ui.output_plot("single_model_spread_plot", height="340px"),
                        full_screen=True,
                    ),
                    col_widths=(12,),
                ),
            )

        if mode != "batch":
            return None

        return ui.tags.div(
            ui.layout_columns(
                ui.card(
                    ui.card_header("Prognoseverlauf"),
                    ui.output_plot("batch_timeline_plot", height="350px"),
                    full_screen=True,
                ),
                col_widths=(12,),
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Höchste Unsicherheit"),
                    ui.output_plot("sigma_rank_plot", height="340px"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Modellvergleich nach Fehler"),
                    ui.output_plot("batch_model_quality_plot", height="340px"),
                    full_screen=True,
                ),
                col_widths=(5, 7),
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Prognose vs. Unsicherheit"),
                    ui.output_plot("uncertainty_scatter_plot", height="320px"),
                ),
                ui.card(
                    ui.card_header("Konfidenzverteilung"),
                    ui.output_plot("confidence_mix_plot", height="240px"),
                ),
                col_widths=(6, 6),
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Prognose-Heatmap"),
                    ui.output_plot("prediction_heatmap_plot", height="300px"),
                ),
                ui.card(
                    ui.card_header("Sigma-Heatmap"),
                    ui.output_plot("sigma_heatmap_plot", height="300px"),
                ),
                col_widths=(6, 6),
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Residuen"),
                    ui.output_plot("residual_scatter_plot", height="280px"),
                ),
                ui.card(
                    ui.card_header("Größte Residuen"),
                    ui.output_plot("abs_residual_rank_plot", height="340px"),
                ),
                col_widths=(4, 8),
            ),
        )

    @render.ui
    def debug_images_stack():
        payload = latest_dashboard_payload()
        if payload is None:
            return ui.p("Keine Debug-Plots verfügbar.")

        labels_and_ids = [
            ("Luftschall Kern", "airborne_debug_core_image"),
            ("Luftschall Gepolstert", "airborne_debug_padded_image"),
            ("Körperschall Kern", "structure_debug_core_image"),
            ("Körperschall Gepolstert", "structure_debug_padded_image"),
        ]

        blocks: list[Any] = []
        for label, image_id in labels_and_ids:
            blocks.append(
                ui.tags.div(
                    ui.tags.div(
                        label,
                        style="font-weight:600; margin-bottom:0.25em"
                    ),
                    ui.output_image(image_id),
                    style="display:block; width:100%; margin-bottom:3rem; overflow:visible;"
                )
            )

        return ui.tags.div(
            *blocks,
            style="display:flex; flex-direction:column; width:100%; gap:5em;"
        )

    # ---------------- Data tables ----------------

    @render.data_frame
    def history_table():
        return _grid(pd.DataFrame(history_rows()))

    @render.data_frame
    def summary_metrics_table():
        return _grid(_summary_metrics_df(latest_dashboard_payload()))

    @render.data_frame
    def prediction_focus_table():
        return _grid(_prediction_focus_df(latest_dashboard_payload(), effective_key()))

    @render.data_frame
    def single_uncertainty_table():
        return _grid(_single_uncertainty_df(latest_dashboard_payload(), effective_key()))

    @render.data_frame
    def artifact_focus_table():
        return _grid(_artifact_focus_df(latest_dashboard_payload(), effective_key()))

    @render.data_frame
    def artifact_paths_table():
        return _grid(_artifact_paths_df(latest_dashboard_payload(), effective_key()))

    @render.data_frame
    def final_predictions_table():
        return _grid(_display_predictions_df(latest_dashboard_payload()))

    @render.data_frame
    def top_uncertainty_table():
        return _grid(_top_uncertainty_df(latest_dashboard_payload()))

    @render.data_frame
    def models_table():
        payload = latest_dashboard_payload()
        return _grid(_metrics_dict_to_df((payload or {}).get("summary", {}).get("models")))

    @render.data_frame
    def modality_fusions_table():
        payload = latest_dashboard_payload()
        return _grid(_metrics_dict_to_df((payload or {}).get("summary", {}).get("modality_fusions")))

    @render.data_frame
    def final_predictions_long_table():
        return _grid(_display_predictions_long_df(latest_dashboard_payload()))

    # ---------------- Plots ----------------

    @render.plot
    def prediction_focus_plot():
        payload = latest_dashboard_payload()
        return make_single_prediction_focus_figure(_selected_prediction_df(payload, effective_key()))

    @render.plot
    def single_sources_plot():
        return make_single_sources_figure(_single_source_df(latest_dashboard_payload(), effective_key()))
    
    @render.plot
    def single_uncertainty_breakdown_plot():
        payload = latest_dashboard_payload()
        if _dashboard_mode(payload) != "single":
            return _message_figure("Nur im Einzelmodus verfügbar.", title="Unsicherheitszerlegung")
        return make_single_uncertainty_breakdown_figure(
            _selected_prediction_df(payload, effective_key())
        )

    @render.plot
    def single_model_spread_plot():
        payload = latest_dashboard_payload()
        if _dashboard_mode(payload) != "single":
            return _message_figure("Nur im Einzelmodus verfügbar.", title="Modellstreuung und Einigung")
        return make_single_model_spread_figure(
            _single_source_df(payload, effective_key()),
            _selected_prediction_df(payload, effective_key()),
        )

    @render.plot
    def batch_timeline_plot():
        payload = latest_dashboard_payload()
        if _dashboard_mode(payload) != "batch":
            return _message_figure("Nur im Batchmodus verfügbar.", title="Prognoseverlauf")
        return make_batch_timeline_figure(_display_predictions_df(payload))

    @render.plot
    def sigma_rank_plot():
        payload = latest_dashboard_payload()
        if _dashboard_mode(payload) != "batch":
            return _message_figure("Nur im Batchmodus verfügbar.", title="Höchste Unsicherheit")
        return make_sigma_rank_figure(_display_predictions_df(payload))
    
    @render.plot
    def batch_model_quality_plot():
        payload = latest_dashboard_payload()
        if _dashboard_mode(payload) != "batch":
            return _message_figure("Nur im Batchmodus verfügbar.", title="Modellvergleich nach Fehler")
        return make_batch_model_quality_figure(_batch_model_quality_df(payload))

    @render.plot
    def uncertainty_scatter_plot():
        payload = latest_dashboard_payload()
        if _dashboard_mode(payload) != "batch":
            return _message_figure("Nur im Batchmodus verfügbar.", title="Prognose vs. Unsicherheit")
        return make_uncertainty_scatter_figure(_display_predictions_df(payload))

    @render.plot
    def residual_scatter_plot():
        payload = latest_dashboard_payload()
        if _dashboard_mode(payload) != "batch":
            return _message_figure("Nur im Batchmodus verfügbar.", title="Residuen")
        return make_residual_scatter_figure(_display_predictions_df(payload))

    @render.plot
    def prediction_heatmap_plot():
        payload = latest_dashboard_payload()
        if _dashboard_mode(payload) != "batch":
            return _message_figure("Nur im Batchmodus verfügbar.", title="Prognose-Heatmap")
        

        plot_df = _display_predictions_df(payload).copy()
        if "y_true" not in plot_df.columns:
            return _message_figure("Für eine Fehler-Heatmap ist kein Ist-Wert verfügbar.")

        plot_df["pred_minus_true_mm"] = (
            pd.to_numeric(plot_df["y_pred"], errors="coerce")
            - pd.to_numeric(plot_df["y_true"], errors="coerce")
        )

        return make_hole_heatmap_figure(
            plot_df,
            value_col="pred_minus_true_mm",
            title="Fehler-Heatmap der Prognose",
            cmap_name="RdBu_r",
            center=0.0,
            value_label="Prognose - Ist [mm]",
        )

    @render.plot
    def sigma_heatmap_plot():
        payload = latest_dashboard_payload()
        if _dashboard_mode(payload) != "batch":
            return _message_figure("Nur im Batchmodus verfügbar.", title="Sigma-Heatmap")
        return make_hole_heatmap_figure(_display_predictions_df(payload), "sigma", "Sigma-Heatmap")

    @render.plot
    def confidence_mix_plot():
        payload = latest_dashboard_payload()
        if _dashboard_mode(payload) != "batch":
            return _message_figure("Nur im Batchmodus verfügbar.", title="Konfidenzverteilung")
        return make_confidence_mix_figure(_display_predictions_df(payload))

    @render.plot
    def abs_residual_rank_plot():
        payload = latest_dashboard_payload()
        if _dashboard_mode(payload) != "batch":
            return _message_figure("Nur im Batchmodus verfügbar.", title="Größte Residuen")
        return make_abs_residual_rank_figure(_display_predictions_df(payload))

    @render.plot
    def airborne_spectrogram_plot():
        payload = latest_dashboard_payload()
        path = _resolve_spectrogram_source(payload, "airborne", effective_key())
        if path is None:
            return _message_figure("Es konnte keine Luftschall-Quelldatei aufgelöst werden.", title="Luftschall-Spektrogramm")
        return make_spectrogram_figure(path, "airborne")

    @render.plot
    def structure_spectrogram_plot():
        payload = latest_dashboard_payload()
        path = _resolve_spectrogram_source(payload, "structure", effective_key())
        if path is None:
            return _message_figure("Es konnte keine Körperschall-Quelldatei aufgelöst werden.", title="Körperschall-Spektrogramm")
        return make_spectrogram_figure(path, "structure")

    # ---------------- Images ----------------

    @render.image
    def airborne_debug_core_image():
        payload = latest_dashboard_payload()
        if payload is None:
            return None
        paths = payload["modalities"]["airborne"].get("debug_core_paths", [])
        path = _existing_path_or_none(paths[0]) if paths else None
        if path is None:
            return None
        return {"src": path, "width": "100%", "height": "auto"}

    @render.image
    def airborne_debug_padded_image():
        payload = latest_dashboard_payload()
        if payload is None:
            return None
        paths = payload["modalities"]["airborne"].get("debug_padded_paths", [])
        path = _existing_path_or_none(paths[0]) if paths else None
        if path is None:
            return None
        return {"src": path, "width": "100%", "height": "auto"}

    @render.image
    def structure_debug_core_image():
        payload = latest_dashboard_payload()
        if payload is None:
            return None
        paths = payload["modalities"]["structure"].get("debug_core_paths", [])
        path = _existing_path_or_none(paths[0]) if paths else None
        if path is None:
            return None
        return {"src": path, "width": "100%", "height": "auto"}

    @render.image
    def structure_debug_padded_image():
        payload = latest_dashboard_payload()
        if payload is None:
            return None
        paths = payload["modalities"]["structure"].get("debug_padded_paths", [])
        path = _existing_path_or_none(paths[0]) if paths else None
        if path is None:
            return None
        return {"src": path, "width": "100%", "height": "auto"}

app = App(app_ui, server)
app.on_shutdown(pool.shutdown)
