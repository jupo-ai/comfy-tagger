import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests


DANBOORU_TAGS_REPO = "jupo-ai/danbooru-characters"
_HF_DATASET_BASE_URL = f"https://huggingface.co/datasets/{DANBOORU_TAGS_REPO}/resolve/main"
_HF_DATASET_API_URL = f"https://huggingface.co/api/datasets/{DANBOORU_TAGS_REPO}/tree/main"
_CACHE_DIR = Path(__file__).parents[3] / "tagger_models" / "danbooru_characters"

_CHARACTER_FILENAME_CANDIDATES = (
    "character_tags.json",
    "characters.json",
    "danbooru_characters.json",
)
_COPYRIGHT_FILENAME_CANDIDATES = (
    "copyright_tags.json",
    "copyrights.json",
    "danbooru_copyrights.json",
)
_META_FILENAME_CANDIDATES = (
    "meta_tags.json",
    "metas.json",
    "danbooru_meta_tags.json",
)


def _normalize_tag(tag: str) -> str:
    return tag.strip().lower().replace(" ", "_")


def _extract_tags(data: Any, category: str) -> set[str]:
    if isinstance(data, list):
        return {str(item) for item in data if str(item).strip()}

    if isinstance(data, dict):
        category_values = data.get(category) or data.get(f"{category}s") or data.get(f"{category}_tags")
        if isinstance(category_values, list):
            return {str(item) for item in category_values if str(item).strip()}
        if all(isinstance(value, bool) for value in data.values()):
            return {str(key) for key, value in data.items() if value}
        return {str(key) for key in data if str(key).strip()}

    return set()


def _download_dataset_file(filename: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / filename
    if path.exists():
        return path

    response = requests.get(f"{_HF_DATASET_BASE_URL}/{filename}", timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


@lru_cache(maxsize=3)
def _remote_json_files() -> tuple[str, ...]:
    response = requests.get(_HF_DATASET_API_URL, timeout=30)
    response.raise_for_status()
    files = []
    for item in response.json():
        path = item.get("path", "")
        if item.get("type") == "file" and path.endswith(".json"):
            files.append(path)
    return tuple(files)


def _find_dataset_file(category: str, candidates: tuple[str, ...]) -> Path:
    for filename in candidates:
        try:
            return _download_dataset_file(filename)
        except Exception:
            continue

    for filename in _remote_json_files():
        lowered = filename.lower()
        if category in lowered:
            return _download_dataset_file(filename)

    raise FileNotFoundError(f"Could not find {category} tags json in {DANBOORU_TAGS_REPO}")


def _load_tag_set(category: str, candidates: tuple[str, ...]) -> set[str]:
    path = _find_dataset_file(category, candidates)
    with open(path, "r", encoding="utf-8") as f:
        return {_normalize_tag(tag) for tag in _extract_tags(json.load(f), category)}


@lru_cache(maxsize=1)
def character_tag_set() -> set[str]:
    return _load_tag_set("character", _CHARACTER_FILENAME_CANDIDATES)


@lru_cache(maxsize=1)
def copyright_tag_set() -> set[str]:
    return _load_tag_set("copyright", _COPYRIGHT_FILENAME_CANDIDATES)


@lru_cache(maxsize=1)
def meta_tag_set() -> set[str]:
    return _load_tag_set("meta", _META_FILENAME_CANDIDATES)


def is_character_tag(tag: str) -> bool:
    return _normalize_tag(tag) in character_tag_set()


def is_copyright_tag(tag: str) -> bool:
    return _normalize_tag(tag) in copyright_tag_set()


def is_meta_tag(tag: str) -> bool:
    return _normalize_tag(tag) in meta_tag_set()


def is_rating_tag(tag: str) -> bool:
    return _normalize_tag(tag).startswith("rating:")
