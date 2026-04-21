from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import librosa.display
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import signal

SPECTROGRAM_CONFIG = {
    "airborne": {
        "nperseg": 4096,
        "noverlap": 4000,
        "max_freq_hz": 20000,
        "n_mels": 128,
        "fmin_hz": 150,
        "top_db": 80,
    },
    "structure": {
        "nperseg": 16384,
        "noverlap": 12000,
        "max_freq_hz": 20000,
        "n_mels": 100,
        "fmin_hz": 150,
        "top_db": 80,
    },
}

COLORS = {
    "final": "#2563eb",
    "airborne": "#0f766e",
    "structure": "#7c3aed",
    "classical": "#64748b",
    "dl": "#f59e0b",
    "sigma": "#93c5fd",
    "risk": "#dc2626",
    "ok": "#16a34a",
    "mid": "#f59e0b",
    "grid": "#d1d5db",
    "text": "#111827",
    "muted": "#4b5563",
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


def _de_modality(value: Any) -> str:
    return MODALITY_DE.get(str(value).strip().lower(), str(value))



def _de_source(value: Any) -> str:
    return SOURCE_DE.get(str(value).strip().lower(), str(value))



def _de_confidence(value: Any) -> str:
    return CONFIDENCE_DE.get(str(value).strip().lower(), str(value))


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _message_figure(
    message: str,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (6.0, 3.2),
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.axis("off")
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.08, top=0.92)
    return fig

def _apply_clean_axis(ax: plt.Axes, *, y_grid: bool = True, x_grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_grid:
        ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.8)
    if x_grid:
        ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)



def _safe_round_df(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False)
    if np.ndim(y) > 1:
        y = np.mean(y, axis=1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    return y, int(sr)



def _load_h5(path: Path) -> tuple[np.ndarray, int]:
    with h5py.File(str(path), "r") as fh:
        y = np.asarray(fh["measurement/data"][:], dtype=np.float32).reshape(-1)
        t = np.asarray(fh["measurement/time_vector"][:], dtype=np.float64).reshape(-1)

    if len(t) < 2:
        raise ValueError(f"Invalid time_vector in {path}")

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError(f"Could not infer sample rate from {path}")

    sr = int(round(1.0 / float(np.median(dt))))
    return y, sr



def load_signal(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()

    if suffix in {".flac", ".wav"}:
        return _load_audio(path)
    if suffix in {".h5", ".hdf5"}:
        return _load_h5(path)

    raise ValueError(f"Nicht unterstützter Spektrogramm-Dateityp: {suffix}")


# ---------------------------------------------------------------------------
# Spectrogram rendering
# ---------------------------------------------------------------------------


def _hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)



def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)



def _build_mel_filterbank(
    freqs_hz: np.ndarray,
    sr: int,
    n_mels: int,
    fmin_hz: float,
    fmax_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    fmax_hz = min(float(fmax_hz), sr / 2.0)
    fmin_hz = max(0.0, float(fmin_hz))

    mel_min = _hz_to_mel(fmin_hz)
    mel_max = _hz_to_mel(fmax_hz)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fb = np.zeros((n_mels, len(freqs_hz)), dtype=np.float32)

    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        if center <= left or right <= center:
            continue

        left_slope = (freqs_hz - left) / (center - left)
        right_slope = (right - freqs_hz) / (right - center)
        fb[i] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    enorm = 2.0 / np.maximum(hz_points[2 : n_mels + 2] - hz_points[:n_mels], 1e-12)
    fb *= enorm[:, None]

    mel_centers_hz = hz_points[1:-1]
    return fb, mel_centers_hz



def _compute_logmel(y: np.ndarray, sr: int, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = min(int(cfg["nperseg"]), max(256, len(y)))
    noverlap = min(int(cfg["noverlap"]), max(128, len(y) // 2))

    freqs_hz, times_s, sxx = signal.spectrogram(
        y,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
        detrend="constant",
    )

    power_spec = sxx**2
    fmin_hz = float(cfg["fmin_hz"])
    fmax_hz = min(float(cfg["max_freq_hz"]), sr / 2.0)

    band_mask = (freqs_hz >= fmin_hz) & (freqs_hz <= fmax_hz)
    freqs_band_hz = freqs_hz[band_mask]
    power_band = power_spec[band_mask, :]

    if len(freqs_band_hz) < 4:
        raise ValueError("Zu wenige Frequenz-Bins für das Log-Mel-Rendering verfügbar.")

    n_freq_bins = len(freqs_band_hz)
    n_mels_eff = max(4, min(int(cfg["n_mels"]), n_freq_bins // 2))

    mel_fb, mel_centers_hz = _build_mel_filterbank(
        freqs_hz=freqs_band_hz,
        sr=sr,
        n_mels=n_mels_eff,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
    )

    mel_power = mel_fb @ power_band
    mel_power = np.maximum(mel_power, 1e-12)

    logmel_db = 10.0 * np.log10(mel_power)
    peak_db = float(np.max(logmel_db))
    floor_db = peak_db - float(cfg["top_db"])
    logmel_db = np.clip(logmel_db, floor_db, peak_db)

    return mel_centers_hz, times_s, logmel_db



def make_spectrogram_figure(file_path: str | Path, modality: str) -> Figure:
    modality = str(modality).strip().lower()
    if modality not in SPECTROGRAM_CONFIG:
        return _message_figure(f"Nicht unterstützte Modalität: {modality}")

    try:
        cfg = SPECTROGRAM_CONFIG[modality]
        y, sr = load_signal(file_path)
        mel_freqs_hz, times_s, logmel_db = _compute_logmel(y=y, sr=sr, cfg=cfg)
    except Exception as exc:
        return _message_figure(str(exc))

    fig, ax = plt.subplots(figsize=(10.8, 4.5), dpi=160)

    hop_length = int(cfg["nperseg"]) - int(cfg["noverlap"])

    img = librosa.display.specshow(
        logmel_db,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=hop_length,
        fmin=float(cfg["fmin_hz"]),
        fmax=min(float(cfg["max_freq_hz"]), sr / 2.0),
        cmap="magma",
        ax=ax,
    )

    ax.set_xlabel("Zeit [s]", fontsize=10)
    ax.set_ylabel("Mel-Frequenz", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label("Log-Mel-Leistung [dB]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.16, top=0.96)
    return fig

# ---------------------------------------------------------------------------
# Dashboard dataframe helpers
# ---------------------------------------------------------------------------


def _parse_step_hole(record_name: Any) -> tuple[float | None, str | None]:
    text = str(record_name or "")
    step_match = re.search(r"step=(\d+)", text)
    hole_match = re.search(r"hole=([A-Za-z]\d+)", text)
    step = float(step_match.group(1)) if step_match else None
    hole = hole_match.group(1).upper() if hole_match else None
    return step, hole



def predictions_rows_to_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).copy()

    for col in (
        "y_pred",
        "sigma",
        "depth_mm",
        "y_true",
        "residual_mm",
        "abs_residual_mm",
        "z_abs",
        "sigma_airborne_mm",
        "sigma_structure_mm",
        "sigma_between_modalities_mm",
        "validation_mae",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "record_name" not in df.columns:
        df["record_name"] = [f"row_{i + 1:03d}" for i in range(len(df))]

    parsed = df["record_name"].apply(_parse_step_hole)
    df["step_idx"] = [item[0] for item in parsed]
    df["hole_id"] = [item[1] for item in parsed]

    if "y_pred" in df.columns:
        rounded = np.round(pd.to_numeric(df["y_pred"], errors="coerce") / 0.1) * 0.1
        df["rounded_step_mm"] = rounded.clip(lower=0.1, upper=1.0)

    if "confidence_label" in df.columns:
        df["confidence_label"] = df["confidence_label"].astype(str)
    elif "sigma" in df.columns:
        sigma = pd.to_numeric(df["sigma"], errors="coerce")
        df["confidence_label"] = np.where(
            sigma < 0.05,
            "high",
            np.where(sigma < 0.15, "medium", "low"),
        )

    return _safe_round_df(df)



def modality_prediction_rows_to_df(payload: dict[str, Any] | None, selected_key: str | None) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    mode = str(payload.get("run", {}).get("mode") or "")

    for modality in ("airborne", "structure"):
        mod_payload = payload.get("modalities", {}).get(modality, {}) or {}
        for source_name, key in (
            ("classical", "classical_predictions"),
            ("dl", "dl_predictions"),
            ("fusion", "fusion_predictions"),
        ):
            pred_rows = mod_payload.get(key) or []
            if not pred_rows:
                continue

            chosen: dict[str, Any] | None = None
            if mode == "single":
                chosen = pred_rows[0]
            else:
                for row in pred_rows:
                    if str(row.get("record_name")) == str(selected_key):
                        chosen = row
                        break
                if chosen is None:
                    chosen = pred_rows[0]

            rows.append(
                {
                    "modality": modality,
                    "source": source_name,
                    "record_name": chosen.get("record_name"),
                    "y_pred": pd.to_numeric(chosen.get("y_pred"), errors="coerce"),
                    "sigma": pd.to_numeric(chosen.get("sigma"), errors="coerce"),
                }
            )

    return _safe_round_df(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Dashboard plots
# ---------------------------------------------------------------------------


def make_single_prediction_focus_figure(df: pd.DataFrame) -> Figure:
    if df.empty:
        return _message_figure("Keine Einzelprognose verfügbar.", title="Prognosefokus")

    row = df.iloc[0]
    pred = float(row.get("y_pred", np.nan))
    sigma = float(row.get("sigma", np.nan)) if pd.notna(row.get("sigma")) else 0.0
    actual = row.get("y_true")

    fig, ax = plt.subplots(figsize=(8.4, 4.1), dpi=150)

    for step in np.arange(0.1, 1.01, 0.1):
        ax.axvline(step, color=COLORS["grid"], linewidth=0.8, zorder=0)

    legend_handles = []

    if np.isfinite(sigma) and sigma > 0:
        ax.axvspan(pred - 2.0 * sigma, pred + 2.0 * sigma, color="#dbeafe", alpha=0.9, zorder=1)
        ax.axvspan(pred - sigma, pred + sigma, color=COLORS["sigma"], alpha=0.95, zorder=2)
        legend_handles.extend([
            Patch(facecolor=COLORS["sigma"], edgecolor="none", label="±1σ-Bereich"),
            Patch(facecolor="#dbeafe", edgecolor="none", label="±2σ-Bereich"),
        ])

    ax.scatter([pred], [0], s=160, color=COLORS["final"], zorder=3)
    legend_handles.insert(
        0,
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["final"], markersize=8, label="Prognose"),
    )

    if pd.notna(actual):
        ax.axvline(float(actual), color=COLORS["ok"], linestyle="--", linewidth=2.0, zorder=3)
        legend_handles.insert(
            1,
            Line2D([0], [0], color=COLORS["ok"], linestyle="--", linewidth=2.0, label="Ist-Wert"),
        )

    span = max(0.18, 3.0 * sigma)
    xmin = max(0.05, pred - span)
    xmax = min(1.05, pred + span)
    if xmax - xmin < 0.35:
        pad = 0.5 * (0.35 - (xmax - xmin))
        xmin = max(0.05, xmin - pad)
        xmax = min(1.05, xmax + pad)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("Tiefe [mm]")
    _apply_clean_axis(ax, x_grid=True)

    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.82)
    return fig

def make_single_sources_figure(df: pd.DataFrame) -> Figure:
    if df.empty:
        return _message_figure("Keine Modalitäts- oder Quellenprognosen verfügbar.")

    plot_df = df.dropna(subset=["y_pred"]).copy()
    if plot_df.empty:
        return _message_figure("Keine numerischen Modalitätsprognosen verfügbar.")

    plot_df["label"] = plot_df["modality"].map(_de_modality) + " · " + plot_df["source"].map(_de_source).str.upper()
    colors = []
    for _, row in plot_df.iterrows():
        if row["source"] == "fusion":
            colors.append(COLORS[row["modality"]])
        elif row["source"] == "classical":
            colors.append(COLORS["classical"])
        else:
            colors.append(COLORS["dl"])

    fig, ax = plt.subplots(figsize=(8.2, 3.5), dpi=150)
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["y_pred"], color=colors)

    sigma = pd.to_numeric(plot_df.get("sigma", pd.Series(0.0, index=plot_df.index)), errors="coerce").fillna(0.0)
    if (sigma > 0).any():
        ax.errorbar(plot_df["y_pred"], y, xerr=sigma, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)

    sigma = pd.to_numeric(
        plot_df.get("sigma", pd.Series(0.0, index=plot_df.index)),
        errors="coerce"
    ).fillna(0.0)

    xmax = max(1.05, float((plot_df["y_pred"] + sigma).max()) + 0.06)
    ax.set_xlim(0.0, xmax)

    for idx, (value, err) in enumerate(zip(plot_df["y_pred"], sigma)):
        value = float(value)
        err = float(err)

        x_text = value + err + 0.012
        ha = "left"

        if x_text > xmax - 0.015:
            x_text = value - err - 0.012
            ha = "right"

        ax.text(
            x_text,
            idx,
            f"{value:.4f}",
            va="center",
            ha=ha,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
        )

    ax.set_yticks(y, plot_df["label"])
    ax.set_xlabel("Vorhergesagte Tiefe [mm]")
    _apply_clean_axis(ax)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.18, top=0.96)
    return fig

def _source_color(source: str, modality: str | None = None) -> str:
    source = str(source).strip().lower()
    modality = str(modality).strip().lower() if modality is not None else ""

    if source == "fusion" and modality in {"airborne", "structure"}:
        return COLORS[modality]
    if source == "classical":
        return COLORS["classical"]
    if source == "dl":
        return COLORS["dl"]
    return COLORS["final"]


def make_single_uncertainty_breakdown_figure(df: pd.DataFrame) -> Figure:
    if df.empty:
        return _message_figure("Keine Einzelprognose für die Unsicherheitszerlegung verfügbar.")

    row = df.iloc[0]
    plot_df = pd.DataFrame(
        {
            "Komponente": ["Luftschall", "Körperschall", "Zwischen Modalitäten"],
            "Wert": [
                pd.to_numeric(row.get("sigma_airborne_mm"), errors="coerce"),
                pd.to_numeric(row.get("sigma_structure_mm"), errors="coerce"),
                pd.to_numeric(row.get("sigma_between_modalities_mm"), errors="coerce"),
            ],
        }
    ).dropna(subset=["Wert"])

    if plot_df.empty:
        return _message_figure("Keine aufgeschlüsselten Unsicherheitsanteile verfügbar.")

    color_map = {
        "Luftschall": COLORS["airborne"],
        "Körperschall": COLORS["structure"],
        "Zwischen Modalitäten": COLORS["final"],
    }
    colors = [color_map[v] for v in plot_df["Komponente"]]

    total_sigma = pd.to_numeric(row.get("sigma"), errors="coerce")

    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=150)
    bars = ax.bar(plot_df["Komponente"], plot_df["Wert"], color=colors, width=0.62)

    for bar, value in zip(bars, plot_df["Wert"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{float(value):.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if pd.notna(total_sigma):
        ax.axhline(float(total_sigma), color=COLORS["muted"], linestyle="--", linewidth=1.2)

    ax.set_ylabel("Sigma-Anteil [mm]")
    _apply_clean_axis(ax, y_grid=True, x_grid=False)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.22, top=0.94)
    return fig


def make_single_model_spread_figure(
    source_df: pd.DataFrame,
    focus_df: pd.DataFrame | None = None,
) -> Figure:
    if source_df.empty:
        return _message_figure("Keine Einzelquellen-Prognosen für die Modellstreuung verfügbar.")

    plot_df = source_df.dropna(subset=["y_pred"]).copy()
    if plot_df.empty:
        return _message_figure("Keine numerischen Einzelquellen-Prognosen verfügbar.")

    final_pred = np.nan
    if focus_df is not None and not focus_df.empty:
        final_pred = pd.to_numeric(focus_df.iloc[0].get("y_pred"), errors="coerce")

    plot_df["label"] = plot_df["modality"].map(_de_modality) + " · " + plot_df["source"].map(_de_source).str.upper()

    if pd.notna(final_pred):
        plot_df["delta_mm"] = pd.to_numeric(plot_df["y_pred"], errors="coerce") - float(final_pred)
        value_col = "delta_mm"
        xlabel = "Abweichung zur finalen Fusion [mm]"
    else:
        value_col = "y_pred"
        xlabel = "Vorhergesagte Tiefe [mm]"

    plot_df = plot_df.sort_values(value_col, ascending=True)

    colors = []
    for _, row in plot_df.iterrows():
        if row["source"] == "fusion":
            colors.append(COLORS[row["modality"]])
        elif row["source"] == "classical":
            colors.append(COLORS["classical"])
        else:
            colors.append(COLORS["dl"])

    values = pd.to_numeric(plot_df[value_col], errors="coerce").to_numpy(dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = max(vmax - vmin, 1e-6)
    pad = 0.06 * span

    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=150)
    bars = ax.barh(plot_df["label"], plot_df[value_col], color=colors)

    if value_col == "delta_mm":
        ax.axvline(0.0, color="black", linewidth=1.2)

    # Slightly wider limits so text has breathing room
    ax.set_xlim(vmin - pad, vmax + pad)

    for bar, value in zip(bars, values):
        y = bar.get_y() + bar.get_height() / 2.0
        label_text = f"{value:+.4f}" if value_col == "delta_mm" else f"{value:.4f}"

        # Put the value inside the bar, away from the y-axis
        if value_col == "delta_mm":
            if value < 0:
                x_text = value + pad * 0.35
                ha = "left"
            else:
                x_text = value - pad * 0.35
                ha = "right"
        else:
            x_text = value - pad * 0.35 if value > 0 else value + pad * 0.35
            ha = "right" if value > 0 else "left"

        ax.text(
            x_text,
            y,
            label_text,
            va="center",
            ha=ha,
            fontsize=8,
            clip_on=True,
        )

    ax.set_xlabel(xlabel)
    _apply_clean_axis(ax, y_grid=True, x_grid=False)
    fig.subplots_adjust(left=0.42, right=0.98, bottom=0.18, top=0.96)
    return fig

def make_batch_timeline_figure(df: pd.DataFrame) -> Figure:
    if df.empty:
        return _message_figure("Keine Batch-Prognosen verfügbar.")

    plot_df = df.copy()
    if "step_idx" in plot_df.columns and plot_df["step_idx"].notna().any():
        plot_df["step_idx"] = pd.to_numeric(plot_df["step_idx"], errors="coerce")
        plot_df = plot_df.sort_values(["step_idx", "record_name"], na_position="last")
        fallback = pd.Series(np.arange(1, len(plot_df) + 1, dtype=float), index=plot_df.index)
        x = plot_df["step_idx"].where(plot_df["step_idx"].notna(), fallback).to_numpy(dtype=float)
        xlabel = "Schritt"
    else:
        plot_df = plot_df.sort_values("record_name")
        x = np.arange(1, len(plot_df) + 1, dtype=float)
        xlabel = "Prognosereihenfolge"

    y = pd.to_numeric(plot_df["y_pred"], errors="coerce").to_numpy(dtype=float)
    sigma = pd.to_numeric(plot_df.get("sigma", pd.Series(0.0, index=plot_df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=150)

    ax.plot(x, y, color=COLORS["final"], marker="o", linewidth=1.9, markersize=4.4, zorder=3)

    legend_handles: list[Any] = [
        Line2D([0], [0], color=COLORS["final"], marker="o", linewidth=1.8, markersize=4.5, label="Prognoseverlauf")
    ]

    finite_blocks: list[np.ndarray] = []
    if np.isfinite(y).any():
        finite_blocks.append(y[np.isfinite(y)])

    if np.isfinite(sigma).any() and np.nanmax(sigma) > 0:
        lower = y - sigma
        upper = y + sigma
        ax.fill_between(x, lower, upper, color=COLORS["sigma"], alpha=0.35, zorder=1)
        legend_handles.append(Patch(facecolor=COLORS["sigma"], edgecolor="none", alpha=0.35, label="±1σ-Band"))
        if np.isfinite(lower).any():
            finite_blocks.append(lower[np.isfinite(lower)])
        if np.isfinite(upper).any():
            finite_blocks.append(upper[np.isfinite(upper)])

    if "y_true" in plot_df.columns:
        y_true = pd.to_numeric(plot_df["y_true"], errors="coerce")
        true_mask = y_true.notna().to_numpy(dtype=bool)
        if true_mask.any():
            y_true_vals = y_true.loc[true_mask].to_numpy(dtype=float)
            ax.scatter(
                x[true_mask],
                y_true_vals,
                marker="x",
                s=34,
                linewidths=1.2,
                color=COLORS["muted"],
                alpha=0.40,
                zorder=2.5,
            )
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker="x",
                    color=COLORS["muted"],
                    linestyle="None",
                    markersize=6,
                    markeredgewidth=1.2,
                    alpha=0.40,
                    label="Ist-Wert",
                )
            )
            if np.isfinite(y_true_vals).any():
                finite_blocks.append(y_true_vals[np.isfinite(y_true_vals)])

    conf_series = plot_df.get("confidence_label", pd.Series("", index=plot_df.index)).astype(str)
    low_mask = conf_series.eq("low").to_numpy(dtype=bool)
    if low_mask.any():
        ax.scatter(x[low_mask], y[low_mask], color=COLORS["risk"], s=42, zorder=4)
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["risk"], markersize=6, label="Niedrige Konfidenz")
        )

    if finite_blocks:
        all_vals = np.concatenate(finite_blocks)
        ymin = float(np.nanmin(all_vals))
        ymax = float(np.nanmax(all_vals))
        pad = 0.08 if np.isclose(ymin, ymax) else max(0.03, 0.08 * (ymax - ymin))
        ax.set_ylim(max(0.0, ymin - pad), min(1.05, ymax + pad))

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Vorhergesagte Tiefe [mm]")

    if len(x) > 0:
        if len(x) <= 20:
            ax.set_xticks(x)
        else:
            step = max(1, len(x) // 10)
            ax.set_xticks(x[::step])

    _apply_clean_axis(ax)

    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=min(4, len(legend_handles)),
        handlelength=1.3,
        handletextpad=0.5,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.80)
    return fig

def make_sigma_rank_figure(df: pd.DataFrame) -> Figure:
    if df.empty or "sigma" not in df.columns:
        return _message_figure("Keine Unsicherheitswerte verfügbar.")

    plot_df = df[["record_name", "sigma"]].copy()
    plot_df["sigma"] = pd.to_numeric(plot_df["sigma"], errors="coerce")
    plot_df = plot_df.dropna().sort_values("sigma", ascending=False).head(10)
    if plot_df.empty:
        return _message_figure("Keine numerischen Unsicherheitswerte verfügbar.")

    plot_df = plot_df.sort_values("sigma", ascending=True)
    colors = [COLORS["risk"] if i >= len(plot_df) - 3 else COLORS["sigma"] for i in range(len(plot_df))]

    fig, ax = plt.subplots(figsize=(6.3, 3.5), dpi=150)
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["sigma"], color=colors)
    ax.set_yticks(y, plot_df["record_name"])
    ax.set_xlabel("Sigma [mm]")
    _apply_clean_axis(ax)
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.18, top=0.96)
    return fig

def make_batch_model_quality_figure(df: pd.DataFrame) -> Figure:
    if df.empty:
        return _message_figure("Keine aggregierten Modellkennzahlen verfügbar.")

    plot_df = df.copy()
    required = {"label", "mae_mm"}
    if not required.issubset(plot_df.columns):
        return _message_figure("Für den Modellvergleich fehlen aggregierte Kennzahlen.")

    plot_df["mae_mm"] = pd.to_numeric(plot_df["mae_mm"], errors="coerce")
    plot_df["spread_mm"] = pd.to_numeric(
        plot_df.get("spread_mm", pd.Series(0.0, index=plot_df.index)),
        errors="coerce",
    ).fillna(0.0)
    plot_df = plot_df.dropna(subset=["mae_mm"])

    if plot_df.empty:
        return _message_figure("Keine numerischen Modellkennzahlen verfügbar.")

    color_map = {
        "classical": COLORS["classical"],
        "dl": COLORS["dl"],
        "airborne": COLORS["airborne"],
        "structure": COLORS["structure"],
        "final": COLORS["final"],
        "muted": COLORS["muted"],
    }
    colors = [color_map.get(str(v).strip().lower(), COLORS["muted"]) for v in plot_df.get("color_key", "muted")]

    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=150)

    x = np.arange(len(plot_df))
    bars = ax.bar(
        x,
        plot_df["mae_mm"],
        color=colors,
        width=0.66,
        zorder=2,
    )

    ax.errorbar(
        x,
        plot_df["mae_mm"],
        yerr=plot_df["spread_mm"],
        fmt="none",
        ecolor=COLORS["text"],
        elinewidth=1.0,
        capsize=4,
        zorder=3,
    )

    ymax = float((plot_df["mae_mm"] + plot_df["spread_mm"]).max())
    ymax = max(ymax * 1.22, 0.01)

    for xi, (_, row) in zip(x, plot_df.iterrows()):
        ax.text(
            xi,
            float(row["mae_mm"]) + float(row["spread_mm"]) + ymax * 0.02,
            f"{float(row['mae_mm']):.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=18, ha="right")
    ax.set_ylabel("MAE [mm]")
    ax.set_xlabel("Modell")
    ax.set_ylim(0.0, ymax)

    _apply_clean_axis(ax, y_grid=True, x_grid=False)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.28, top=0.96)
    return fig

def make_confidence_mix_figure(df: pd.DataFrame) -> Figure:
    if df.empty or "confidence_label" not in df.columns:
        return _message_figure("Keine Konfidenzstufen verfügbar.")

    counts = df["confidence_label"].astype(str).value_counts().reindex(["high", "medium", "low"], fill_value=0)
    colors = [COLORS["ok"], COLORS["mid"], COLORS["risk"]]

    fig, ax = plt.subplots(figsize=(5.3, 3.1), dpi=150)
    x_labels = [_de_confidence(v) for v in counts.index]
    ax.bar(x_labels, counts.values, color=colors, width=0.62)
    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.05, str(int(value)), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Anzahl")
    _apply_clean_axis(ax)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.96)
    return fig

def make_uncertainty_scatter_figure(df: pd.DataFrame) -> Figure:
    required = {"y_pred", "sigma"}
    if df.empty or not required.issubset(df.columns):
        return _message_figure("Spalten für Prognose oder Sigma fehlen.")

    plot_df = df.copy()
    plot_df["y_pred"] = pd.to_numeric(plot_df["y_pred"], errors="coerce")
    plot_df["sigma"] = pd.to_numeric(plot_df["sigma"], errors="coerce")
    plot_df = plot_df.dropna(subset=["y_pred", "sigma"])
    if plot_df.empty:
        return _message_figure("Keine numerischen Werte verfügbar.")

    color_map = {"high": COLORS["ok"], "medium": COLORS["mid"], "low": COLORS["risk"]}
    plot_df["confidence_label"] = plot_df.get("confidence_label", "").astype(str)
    colors = [color_map.get(v, COLORS["final"]) for v in plot_df["confidence_label"]]

    fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=150)

    ax.scatter(
        plot_df["y_pred"],
        plot_df["sigma"],
        c=colors,
        s=46,
        alpha=0.82,
        edgecolors="none",
    )

    ax.set_xlabel("Vorhergesagte Tiefe [mm]")
    ax.set_ylabel("Sigma [mm]")
    _apply_clean_axis(ax)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="none",
               markerfacecolor=COLORS["ok"], markersize=6, label="Hohe Konfidenz"),
        Line2D([0], [0], marker="o", linestyle="None", color="none",
               markerfacecolor=COLORS["mid"], markersize=6, label="Mittlere Konfidenz"),
        Line2D([0], [0], marker="o", linestyle="None", color="none",
               markerfacecolor=COLORS["risk"], markersize=6, label="Niedrige Konfidenz"),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=7,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.03),
        ncol=3,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.76)
    return fig

def make_residual_scatter_figure(df: pd.DataFrame) -> Figure:
    required = {"y_true", "residual_mm"}
    if df.empty or not required.issubset(df.columns):
        return _message_figure("Für die Residuenanalyse ist kein Ist-Wert verfügbar.")

    plot_df = df.copy()
    plot_df["y_true"] = pd.to_numeric(plot_df["y_true"], errors="coerce")
    plot_df["residual_mm"] = pd.to_numeric(plot_df["residual_mm"], errors="coerce")
    plot_df = plot_df.dropna(subset=["y_true", "residual_mm"])
    if plot_df.empty:
        return _message_figure("Keine numerischen Residualwerte verfügbar.")

    fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=150)
    ax.axhline(0.0, color="black", linewidth=1.2)
    ax.scatter(plot_df["y_true"], plot_df["residual_mm"], color=COLORS["final"], s=50, alpha=0.9)
    ax.set_xlabel("Ist-Tiefe [mm]")
    ax.set_ylabel("Residuum [mm]")
    _apply_clean_axis(ax)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.22, top=0.96)
    return fig

def make_abs_residual_rank_figure(df: pd.DataFrame) -> Figure:
    required = {"record_name", "abs_residual_mm"}
    if df.empty or not required.issubset(df.columns):
        return _message_figure("Absolute Residuen sind nicht verfügbar.")

    plot_df = df[["record_name", "abs_residual_mm"]].copy()
    plot_df["abs_residual_mm"] = pd.to_numeric(plot_df["abs_residual_mm"], errors="coerce")
    plot_df = plot_df.dropna().sort_values("abs_residual_mm", ascending=False).head(10)
    if plot_df.empty:
        return _message_figure("Keine numerischen Residuen verfügbar.")

    plot_df = plot_df.sort_values("abs_residual_mm", ascending=True)

    plot_df["label"] = (
        plot_df["record_name"]
        .astype(str)
        .str.replace("step=", "s", regex=False)
        .str.replace("__hole=", " · ", regex=False)
    )

    fig_h = max(3.4, 0.42 * len(plot_df) + 0.6)
    fig, ax = plt.subplots(figsize=(7.2, fig_h), dpi=150)

    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["abs_residual_mm"], color=COLORS["risk"])
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=9)
    ax.tick_params(axis="y", pad=4)
    ax.set_xlabel("Absolutes Residuum [mm]")

    _apply_clean_axis(ax)
    fig.subplots_adjust(left=0.26, right=0.98, bottom=0.16, top=0.96)
    return fig

def _hole_grid(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, list[str], list[str]] | None:
    if df.empty or "hole_id" not in df.columns or value_col not in df.columns:
        return None

    plot_df = df[["hole_id", value_col]].copy()
    plot_df = plot_df.dropna(subset=["hole_id"])
    if plot_df.empty:
        return None

    plot_df["row_letter"] = plot_df["hole_id"].astype(str).str.extract(r"([A-Za-z])", expand=False).str.upper()
    plot_df["col_number"] = pd.to_numeric(plot_df["hole_id"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["row_letter", "col_number", value_col])
    if plot_df.empty:
        return None

    rows = sorted(plot_df["row_letter"].unique())
    cols = sorted(int(c) for c in plot_df["col_number"].unique())
    arr = np.full((len(rows), len(cols)), np.nan, dtype=float)

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            mask = (plot_df["row_letter"] == row) & (plot_df["col_number"] == col)
            if mask.any():
                arr[i, j] = float(plot_df.loc[mask, value_col].mean())

    return arr, rows, [str(c) for c in cols]



def make_hole_heatmap_figure(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    cmap_name: str = "RdBu_r",
    center: float | None = None,
    value_label: str | None = None,
) -> Figure:
    grid = _hole_grid(df, value_col)
    if grid is None:
        return _message_figure("Kein bohrungsbasiertes Batch-Layout verfügbar.")

    arr, rows, cols = grid
    cmap = plt.get_cmap(cmap_name)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return _message_figure("Keine numerischen Werte für die Heatmap verfügbar.")

    if center is not None:
        span = float(np.nanmax(np.abs(finite - center)))
        if np.isclose(span, 0.0):
            span = 1e-9
        vmin = center - span
        vmax = center + span
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    else:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-9
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(5.8, 3.9), dpi=150)
    im = ax.imshow(arr, cmap=cmap, aspect="auto", interpolation="nearest", norm=norm)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Spalte")
    ax.set_ylabel("Zeile")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isfinite(arr[i, j]):
                continue

            rgba = cmap(norm(arr[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = "black" if luminance > 0.63 else "white"

            if value_col == "confidence_score":
                text_value = f"{arr[i, j]:.0%}"
            elif center is not None:
                text_value = f"{arr[i, j]:+.3f}"
            else:
                text_value = f"{arr[i, j]:.3f}"

            bbox_fc = (1, 1, 1, 0.18) if txt_color == "black" else (0, 0, 0, 0.18)
            ax.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                color=txt_color,
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.14", fc=bbox_fc, ec="none"),
            )

    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.ax.set_ylabel(value_label or value_col.replace("_", " "))
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.14, right=0.92, bottom=0.20, top=0.96)
    return fig