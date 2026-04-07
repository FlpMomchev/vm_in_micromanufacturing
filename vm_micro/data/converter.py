"""vm_micro.data.converter
~~~~~~~~~~~~~~~~~~~~~~~~~
Convert QASS ``.000`` buffer files to the HDF5 layout expected by the rest
of the pipeline (``measurement/data``, ``measurement/time_vector``, plus
metadata attributes).

The conversion is intentionally kept as a thin, single-file operation so it
can be called both from the dashboard watcher (automatic) and from a
standalone CLI script.

Requires the proprietary ``qass`` package (``qass.tools.analyzer.buffer_parser``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

BUFFER_SUFFIX = ".000"

# ------------------------------------------------------------------ #
# Public helpers
# ------------------------------------------------------------------ #


def is_buffer_file(path: Path) -> bool:
    """Return *True* if *path* looks like a QASS buffer file."""
    return path.suffix.lower() == BUFFER_SUFFIX


def h5_target_for(buffer_path: Path) -> Path:
    """Return the ``.h5`` path that a converted buffer would be written to.

    Naming rule: extract the run number from the ``p<NNN>c`` pattern in
    the filename (matching the existing data naming convention).  Fall
    back to the bare stem if no pattern is found.
    """
    match = re.search(r"p(\d+)c", buffer_path.name)
    stem = str(int(match.group(1))) if match else buffer_path.stem
    return buffer_path.with_name(f"{stem}.h5")


# ------------------------------------------------------------------ #
# Core conversion
# ------------------------------------------------------------------ #


def convert_buffer_to_h5(
    src: Path,
    dst: Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert a single QASS ``.000`` buffer file to HDF5.

    Parameters
    ----------
    src:
        Path to the ``.000`` file.
    dst:
        Destination ``.h5`` path.  Derived automatically via
        :func:`h5_target_for` when *None*.
    overwrite:
        When *False* (default) and *dst* already exists, the conversion
        is skipped and the existing path is returned.

    Returns
    -------
    Path to the written (or already-existing) ``.h5`` file.
    """
    # Late import — qass is proprietary and may not be installed in
    # every environment.
    try:
        from qass.tools.analyzer.buffer_parser import Buffer
    except ImportError as exc:
        raise ImportError(
            "The 'qass' package is required for .000 buffer conversion but is "
            "not installed.  Install it or convert files manually before "
            "placing them in the watch directory."
        ) from exc

    src = Path(src).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Buffer file does not exist: {src}")

    if dst is None:
        dst = h5_target_for(src)
    dst = Path(dst).expanduser().resolve()

    if dst.exists() and not overwrite:
        logger.info("Converted file already exists, skipping: %s", dst.name)
        return dst

    # --- read buffer ------------------------------------------------
    with Buffer(str(src)) as buff:
        sampling_rate = buff.metainfo.get("samplert")
        if sampling_rate is None:
            raise ValueError(f"Sampling rate not found in buffer metadata: {src}")

        process_id = buff.process
        production_date = buff.process_date_time
        bit_resolution = buff.bit_resolution
        sample_count = buff.sample_count
        data_mode = buff.datamode.name

        data = buff.get_data()

    data = np.asarray(data, dtype=np.float32)
    time_vector = np.arange(sample_count, dtype=np.float64) / float(sampling_rate)

    # --- write HDF5 -------------------------------------------------
    tmp = dst.with_suffix(".h5.tmp")
    try:
        with h5py.File(tmp, "w") as h5f:
            grp = h5f.create_group("measurement")
            grp.create_dataset(
                "data", data=data, compression="gzip", compression_opts=4, shuffle=True,
            )
            grp.create_dataset(
                "time_vector", data=time_vector, compression="gzip", compression_opts=4,
                shuffle=True,
            )
            grp.attrs["process_id"] = str(process_id)
            grp.attrs["production_date"] = str(production_date)
            grp.attrs["sampling_rate"] = float(sampling_rate)
            grp.attrs["bit_resolution"] = int(bit_resolution)
            grp.attrs["sample_count"] = int(sample_count)
            grp.attrs["data_mode"] = str(data_mode)
            grp.attrs["source_file"] = src.name

        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    logger.info("Converted buffer → HDF5: %s → %s", src.name, dst.name)
    return dst