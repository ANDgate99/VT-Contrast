import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional

logger = logging.getLogger(__name__)

WANDB_RUN_INFO_FILENAME = "wandb_run.json"


def _normalize_directory(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None

    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.parent
    return candidate


def _iter_search_directories(*paths: Optional[str]) -> Iterator[Path]:
    seen: set[Path] = set()
    for raw_path in paths:
        current = _normalize_directory(raw_path)
        while current is not None:
            resolved = current.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved
            if current.parent == current:
                break
            current = current.parent


def _read_run_info_file(path: Path) -> Optional[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        logger.warning("Ignore invalid W&B run info file: %s", path)
        return None

    run_id = payload.get("run_id")
    if not run_id:
        return None
    return payload


def find_wandb_run_info(
    output_dir: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    for directory in _iter_search_directories(resume_from_checkpoint, output_dir):
        run_info_path = directory / WANDB_RUN_INFO_FILENAME
        run_info = _read_run_info_file(run_info_path)
        if run_info is not None:
            run_info.setdefault("source_path", str(run_info_path))
            return run_info

    env_run_id = os.getenv("WANDB_RUN_ID")
    if env_run_id:
        return {
            "run_id": env_run_id,
            "source_path": "env:WANDB_RUN_ID",
        }

    return None


def save_wandb_run_info(
    output_dir: str,
    run_id: str,
    project: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    run_info_path = output_path / WANDB_RUN_INFO_FILENAME
    payload = {
        "run_id": run_id,
        "project": project,
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    tmp_path = run_info_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, run_info_path)

    return str(run_info_path)
