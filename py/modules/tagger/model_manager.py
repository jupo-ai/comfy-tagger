from pathlib import Path
from typing import Protocol

import requests
from tqdm import tqdm

from .registry import get_model_spec


class DownloadProgressLike(Protocol):
    def update_absolute(self, value: int, total: int) -> None:
        ...


def get_model_path(models_dir: Path, model_name: str, ext: str = "onnx") -> Path:
    if ext.startswith("."):
        ext = ext.lstrip(".")
    return Path(models_dir) / f"{model_name}.{ext}"


def get_model_paths(model_name: str, models_dir: Path) -> dict[str, Path]:
    spec = get_model_spec(model_name)
    paths: dict[str, Path] = {}
    for file in spec.files:
        path = get_model_path(models_dir, model_name, file.ext)
        if path.exists() or file.required:
            paths[file.filename] = path
            paths[Path(file.filename).name] = path
    return paths


def download_model(
    model_name: str,
    models_dir: Path,
    progress_bar: DownloadProgressLike | None = None,
) -> bool:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    spec = get_model_spec(model_name)
    base_url = f"https://huggingface.co/{spec.repo_id}/resolve/main"

    download_jobs = []
    for file in spec.files:
        dest = get_model_path(models_dir, model_name, file.ext)
        if not dest.exists():
            download_jobs.append((f"{base_url}/{file.filename}", dest, file.required))

    progress_state = None
    if progress_bar is not None and download_jobs:
        progress_state = {
            "downloaded": 0,
            "total": sum(_get_content_length(url) for url, _, _ in download_jobs),
        }

    for url, dest, required in download_jobs:
        if not _download_file(url, dest, progress_bar, progress_state) and required:
            return False

    return True


def prepare_model(
    model_name: str,
    models_dir: Path,
    progress_bar: DownloadProgressLike | None = None,
) -> dict[str, Path]:
    if not download_model(model_name, models_dir, progress_bar):
        raise RuntimeError(f"Failed to prepare tagger model: {model_name}")
    return get_model_paths(model_name, models_dir)


def _get_content_length(url: str) -> int:
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        return int(response.headers.get("content-length", 0))
    except Exception:
        return 0


def _download_file(
    url: str,
    dest_path: Path,
    progress_bar: DownloadProgressLike | None = None,
    progress_state: dict[str, int] | None = None,
) -> bool:
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with tqdm(
            total=total_size if total_size > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {dest_path.name}",
        ) as pbar:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue

                    f.write(chunk)
                    chunk_size = len(chunk)
                    downloaded += chunk_size
                    pbar.update(chunk_size)
                    if progress_bar is not None:
                        if progress_state is not None and progress_state["total"] > 0:
                            progress_state["downloaded"] += chunk_size
                            current = min(progress_state["total"], progress_state["downloaded"])
                            progress_bar.update_absolute(current, progress_state["total"])
                        elif total_size > 0:
                            current = min(total_size, downloaded)
                            progress_bar.update_absolute(current, total_size)

        return True

    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False
