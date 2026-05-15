import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests


DANBOORU_TAGS_REPO = "jupo-ai/danbooru-tag-names"
_HF_DATASET_BASE_URL = f"https://huggingface.co/datasets/{DANBOORU_TAGS_REPO}/resolve/main"
_HF_DATASET_API_URL = f"https://huggingface.co/api/datasets/{DANBOORU_TAGS_REPO}/tree/main"
_CACHE_DIR = Path(__file__).parents[2] / "data" / "danbooru-tag-names"

_CHARACTER_FILENAME_CANDIDATES = (
    "character_tags.json",
)
_COPYRIGHT_FILENAME_CANDIDATES = (
    "copyright_tags.json",
)
_ARTIST_FILENAME_CANDIDATES = (
    "artist_tags.json",
)
_META_FILENAME_CANDIDATES = (
    "meta_tags.json",
)


def _normalize_tag(tag: str) -> str:
    """タグ照合用の正規化文字列を作ります。

    タグ文字列を受け取り、前後空白除去、小文字化、空白の `_` 置換をした文字列を返します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。

    Returns
    -------
    returns : str
        変換または生成した文字列を返します。
    """
    return tag.strip().lower().replace(" ", "_")


def _extract_tags(data: Any, category: str) -> set[str]:
    """JSON データから指定カテゴリのタグ集合を抽出します。

    JSON として読んだ任意データとカテゴリ名を受け取り、抽出できたタグ文字列の集合を返します。

    Parameters
    ----------
    data : Any
        JSON から読み込んだ任意のデータです。
    category : str
        タグカテゴリ名です。

    Returns
    -------
    returns : set[str]
        処理結果の集合を返します。
    """
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
    """Danbooru タグカタログ用ファイルを取得します。

    ファイル名を受け取り、ローカルキャッシュにあればそのパスを、なければダウンロード後のパスを返します。

    Parameters
    ----------
    filename : str
        取得または登録するファイル名です。

    Returns
    -------
    returns : Path
        対象ファイルまたはディレクトリのパスを返します。
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / filename
    if path.exists():
        return path

    print(f"[TaggerUI] Downloading Danbooru tag names data: {filename} -> {path}", flush=True)
    response = requests.get(f"{_HF_DATASET_BASE_URL}/{filename}", timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


@lru_cache(maxsize=3)
def _remote_json_files() -> tuple[str, ...]:
    """リモートデータセット内の JSON ファイル一覧を取得します。

    引数は受け取らず、Hugging Face API から取得した JSON ファイルパスのタプルを返します。

    Returns
    -------
    returns : tuple[str, ...]
        複数の処理結果をまとめたタプルを返します。
    """
    response = requests.get(_HF_DATASET_API_URL, timeout=30)
    response.raise_for_status()
    files = []
    for item in response.json():
        path = item.get("path", "")
        if item.get("type") == "file" and path.endswith(".json"):
            files.append(path)
    return tuple(files)


def _find_dataset_file(category: str, candidates: tuple[str, ...]) -> Path:
    """カテゴリに対応するデータセットファイルを探します。

    カテゴリ名と候補ファイル名を受け取り、利用できる JSON ファイルのローカルパスを返します。

    Parameters
    ----------
    category : str
        タグカテゴリ名です。
    candidates : tuple[str, ...]
        候補ファイル名のタプルです。

    Returns
    -------
    returns : Path
        対象ファイルまたはディレクトリのパスを返します。

    Raises
    ------
    FileNotFoundError
        必要なデータセットファイルを見つけられない場合に送出されます。
    """
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
    """カテゴリタグ集合を読み込みます。

    カテゴリ名と候補ファイル名を受け取り、正規化済みタグ文字列の集合を返します。

    Parameters
    ----------
    category : str
        タグカテゴリ名です。
    candidates : tuple[str, ...]
        候補ファイル名のタプルです。

    Returns
    -------
    returns : set[str]
        処理結果の集合を返します。
    """
    path = _find_dataset_file(category, candidates)
    with open(path, "r", encoding="utf-8") as f:
        return {_normalize_tag(tag) for tag in _extract_tags(json.load(f), category)}


@lru_cache(maxsize=1)
def character_tag_set() -> set[str]:
    """キャラクタータグ集合を返します。

    Returns
    -------
    returns : set[str]
        処理結果の集合を返します。
    """
    return _load_tag_set("character", _CHARACTER_FILENAME_CANDIDATES)


@lru_cache(maxsize=1)
def copyright_tag_set() -> set[str]:
    """版権タグ集合を返します。

    Returns
    -------
    returns : set[str]
        処理結果の集合を返します。
    """
    return _load_tag_set("copyright", _COPYRIGHT_FILENAME_CANDIDATES)


@lru_cache(maxsize=1)
def artist_tag_set() -> set[str]:
    """アーティストタグ集合を返します。

    Returns
    -------
    returns : set[str]
        処理結果の集合を返します。
    """
    return _load_tag_set("artist", _ARTIST_FILENAME_CANDIDATES)


@lru_cache(maxsize=1)
def meta_tag_set() -> set[str]:
    """メタタグ集合を返します。

    Returns
    -------
    returns : set[str]
        処理結果の集合を返します。
    """
    return _load_tag_set("meta", _META_FILENAME_CANDIDATES)


@lru_cache(maxsize=1)
def rating_tag_set() -> set[str]:
    """レーティングタグ集合を返します。
    
    Returns
    -------
    returns : set[str]
        処理結果の集合を返します。
    """
    ratings = (
        "explicit", 
        "questionable", 
        "sensitive", 
        "general", 
        "rating:explicit", 
        "rating:questionable", 
        "rating:sensitive", 
        "rating:general", 
        "safe", 
        "rating:safe", 
    )
    return set(ratings)
    


def is_character_tag(tag: str) -> bool:
    """タグがキャラクタータグか判定します。

    タグ文字列を受け取り、カタログ上のキャラクタータグなら `True` を返します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    return _normalize_tag(tag) in character_tag_set()


def is_copyright_tag(tag: str) -> bool:
    """タグが版権タグか判定します。

    タグ文字列を受け取り、カタログ上の版権タグなら `True` を返します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    return _normalize_tag(tag) in copyright_tag_set()


def is_artist_tag(tag: str) -> bool:
    """タグがアーティストタグか判定します。

    タグ文字列を受け取り、カタログ上のアーティストタグなら `True` を返します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    return _normalize_tag(tag) in artist_tag_set()


def is_meta_tag(tag: str) -> bool:
    """タグがメタタグか判定します。

    タグ文字列を受け取り、カタログ上のメタタグなら `True` を返します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    return _normalize_tag(tag) in meta_tag_set()


def is_rating_tag(tag: str) -> bool:
    """タグが rating 名前空間のタグか判定します。

    タグ文字列を受け取り、カタログ上のレーティングタグなら `True` を返します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    return _normalize_tag(tag) in rating_tag_set()
