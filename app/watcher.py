from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Iterable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from vm_micro.data.converter import convert_buffer_to_h5, is_buffer_file

from .db import DashboardDB
from .settings import AppSettings, load_settings

logger = logging.getLogger(__name__)

# Airborne files must contain this substring to be accepted (mic-2).
_AIRBORNE_NAME_FILTER = "-2_"


def _normalize_extensions(values: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        item = value.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        normalized.append(item)
    return tuple(dict.fromkeys(normalized))


def _iter_existing_candidate_files(
    root: Path,
    allowed_extensions: tuple[str, ...],
) -> list[Path]:
    if not root.exists():
        return []

    candidates: list[Path] = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_extensions:
            continue
        candidates.append(path)

    return candidates


def bootstrap_latest_files(settings: AppSettings, db: DashboardDB) -> None:
    # Convert any .000 buffer files sitting in the structure folder before
    # scanning for candidates, so the resulting .h5 files are picked up.
    _convert_pending_buffers(settings)

    modality_dirs = {
        "airborne": settings.watch_dir_airborne,
        "structure": settings.watch_dir_structure,
    }

    for modality, modality_dir in modality_dirs.items():
        candidates = _iter_existing_candidate_files(
            modality_dir,
            settings.allowed_extensions,
        )

        # Apply airborne mic-2 filter.
        if modality == "airborne":
            candidates = [p for p in candidates if _AIRBORNE_NAME_FILTER in p.name]

        if not candidates:
            continue

        latest = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        db.upsert_detected_file(latest, modality=modality)
        logger.info(
            "Bootstrapped latest file | modality=%s | file=%s",
            modality,
            latest.name,
        )


def _convert_pending_buffers(settings: AppSettings) -> None:
    """Convert any unconverted .000 files in the structure watch directory."""
    structure_dir = settings.watch_dir_structure
    if not structure_dir.exists():
        return

    for path in structure_dir.iterdir():
        if not path.is_file() or not is_buffer_file(path):
            continue

        # Skip FFT / large files — only convert raw buffers.
        try:
            if path.stat().st_size > settings.buffer_max_size_bytes:
                logger.debug(
                    "Skipped oversized buffer file (FFT): %s (%.1f MB)",
                    path.name,
                    path.stat().st_size / (1024 * 1024),
                )
                continue
        except OSError:
            continue

        try:
            convert_buffer_to_h5(path)
        except ImportError:
            logger.warning(
                "qass package not installed — cannot auto-convert .000 files. "
                "Place pre-converted .h5 files in the watch directory instead."
            )
            return
        except Exception:
            logger.exception("Failed to convert buffer file at startup: %s", path.name)


class LatestFileEventHandler(FileSystemEventHandler):
    """Watch a folder and keep only the newest relevant file marked as latest."""

    def __init__(
        self,
        db: DashboardDB,
        watch_dir_airborne: Path,
        watch_dir_structure: Path,
        allowed_extensions: Iterable[str],
        buffer_extensions: Iterable[str] = (),
        buffer_max_size_bytes: int = 400 * 1024 * 1024,
        settle_time_sec: float = 0.75,
    ) -> None:
        super().__init__()
        self.db = db
        self.watch_dir_airborne = watch_dir_airborne.resolve()
        self.watch_dir_structure = watch_dir_structure.resolve()
        self.allowed_extensions = _normalize_extensions(allowed_extensions)
        self.buffer_extensions = _normalize_extensions(buffer_extensions)
        self.buffer_max_size_bytes = buffer_max_size_bytes
        self.settle_time_sec = float(settle_time_sec)
        self._lock = threading.Lock()

    def _infer_modality(self, path: Path) -> str | None:
        """Infer modality by checking which configured watch dir the file is in."""
        try:
            if path.is_relative_to(self.watch_dir_airborne):
                return "airborne"
            if path.is_relative_to(self.watch_dir_structure):
                return "structure"
        except ValueError:
            pass
        return None

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def _handle_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        src_path = getattr(event, "src_path", None)
        dest_path = getattr(event, "dest_path", None)

        candidate = dest_path or src_path
        if not candidate:
            return

        path = Path(candidate).expanduser().resolve()
        modality = self._infer_modality(path)
        if modality is None:
            return

        # --- auto-convert .000 buffer files --------------------------
        if path.suffix.lower() in self.buffer_extensions:
            if modality != "structure":
                return
            with self._lock:
                if not self._wait_until_readable(path):
                    logger.debug("Skipped unreadable buffer file: %s", path)
                    return

                # Skip FFT / large files.
                try:
                    if path.stat().st_size > self.buffer_max_size_bytes:
                        logger.debug(
                            "Skipped oversized buffer file (FFT): %s (%.1f MB)",
                            path.name,
                            path.stat().st_size / (1024 * 1024),
                        )
                        return
                except OSError:
                    return

                try:
                    convert_buffer_to_h5(path)
                except ImportError:
                    logger.warning(
                        "qass package not installed — cannot auto-convert %s",
                        path.name,
                    )
                except Exception:
                    logger.exception("Buffer conversion failed: %s", path.name)
            return

        if path.suffix.lower() not in self.allowed_extensions:
            return

        # --- airborne mic-2 filter -----------------------------------
        if modality == "airborne" and _AIRBORNE_NAME_FILTER not in path.name:
            logger.debug("Skipped non-mic-2 airborne file: %s", path.name)
            return

        with self._lock:
            if not self._wait_until_readable(path):
                logger.debug("Skipped unreadable or unstable file: %s", path)
                return

            try:
                row = self.db.upsert_detected_file(path, modality=modality)
                logger.info(
                    "Latest detected file updated | modality=%s | id=%s | name=%s | status=%s",
                    row["modality"],
                    row["id"],
                    row["file_name"],
                    row["status"],
                )
            except FileNotFoundError:
                logger.debug("File disappeared before registration: %s", path)
            except Exception:
                logger.exception("Failed to register detected file: %s", path)

    def _wait_until_readable(self, path: Path) -> bool:
        """Wait briefly until the file exists and its size stops changing."""
        deadline = time.monotonic() + max(self.settle_time_sec, 0.0)
        last_size: int | None = None

        while time.monotonic() <= deadline:
            try:
                if not path.is_file():
                    time.sleep(0.1)
                    continue

                current_size = path.stat().st_size

                if last_size is not None and current_size == last_size:
                    with path.open("rb"):
                        return True

                last_size = current_size
            except OSError:
                pass

            time.sleep(0.1)

        try:
            if path.is_file():
                with path.open("rb"):
                    return True
        except OSError:
            return False

        return False


class LatestFileWatcher:
    """Small wrapper around watchdog for the dashboard app."""

    def __init__(self, settings: AppSettings, db: DashboardDB) -> None:
        self.settings = settings
        self.db = db
        self.handler = LatestFileEventHandler(
            db=db,
            watch_dir_airborne=settings.watch_dir_airborne,
            watch_dir_structure=settings.watch_dir_structure,
            allowed_extensions=settings.allowed_extensions,
            buffer_extensions=settings.buffer_extensions,
            buffer_max_size_bytes=settings.buffer_max_size_bytes,
        )
        self.observer = Observer()
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        self.settings.watch_dir_airborne.mkdir(parents=True, exist_ok=True)
        self.settings.watch_dir_structure.mkdir(parents=True, exist_ok=True)

        # Schedule one watch per modality directory (non-recursive, files
        # land directly in the folder).
        self.observer.schedule(
            self.handler, str(self.settings.watch_dir_airborne), recursive=False,
        )
        self.observer.schedule(
            self.handler, str(self.settings.watch_dir_structure), recursive=False,
        )
        self.observer.start()
        self._started = True

        logger.info("Watching airborne: %s", self.settings.watch_dir_airborne)
        logger.info("Watching structure: %s", self.settings.watch_dir_structure)
        logger.info("Allowed extensions: %s", ", ".join(self.settings.allowed_extensions))

    def stop(self) -> None:
        if not self._started:
            return

        self.observer.stop()
        self.observer.join(timeout=5.0)
        self._started = False
        logger.info("Watcher stopped")

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Watcher interrupted by user")
        finally:
            self.stop()


def run_latest_file_watcher() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    settings = load_settings()
    db = DashboardDB(settings.db_path)
    db.init()

    bootstrap_latest_files(settings=settings, db=db)

    watcher = LatestFileWatcher(settings=settings, db=db)
    watcher.run_forever()


if __name__ == "__main__":
    run_latest_file_watcher()