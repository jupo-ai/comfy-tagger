from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import requests

from .match import _split_to_words, _words_to_matcher


TAG_FILTERING_REPO = "alea31415/tag_filtering"
BLACKLIST_FILENAME = "blacklist_tags.txt"
_HF_DATASET_BASE_URL = f"https://huggingface.co/datasets/{TAG_FILTERING_REPO}/resolve/main"
_CACHE_DIR = Path(__file__).parents[2] / "data" / "tag_filtering"


def _download_dataset_file(filename: str) -> Path:
    """タグフィルタリング用データセットファイルを取得します。

    ファイル名を受け取り、ローカルキャッシュに存在すればそのパスを、なければダウンロード後のパスを返します。

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

    print(f"[TaggerUI] Downloading tag filtering data: {filename} -> {path}", flush=True)
    response = requests.get(f"{_HF_DATASET_BASE_URL}/{filename}", timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


@lru_cache(maxsize=1)
def _load_online_blacklist() -> list[str]:
    """オンライン由来のブラックリストタグを読み込みます。

    引数は受け取らず、空行を除いたブラックリスト文字列のリストを返します。

    Returns
    -------
    returns : list[str]
        処理結果のリストを返します。
    """
    with open(_download_dataset_file(BLACKLIST_FILENAME), "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@lru_cache(maxsize=1)
def _online_blacklist_set() -> set[tuple[str, ...]]:
    """ブラックリストを照合用タプル集合へ変換します。

    引数は受け取らず、単語列タプルの集合を返します。

    Returns
    -------
    returns : set[tuple[str, ...]]
        複数の処理結果をまとめたタプルを返します。
    """
    set_ = set()
    for tag in _load_online_blacklist():
        set_ = set_ | _words_to_matcher(_split_to_words(tag))
    return set_


def _is_blacklisted(tag: str, blacklist_set: set[tuple[str, ...]]) -> bool:
    """タグが指定ブラックリスト集合に含まれるか判定します。

    タグ文字列と照合用ブラックリスト集合を受け取り、一致候補があれば `True` を返します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。
    blacklist_set : set[tuple[str, ...]]
        照合用ブラックリスト集合です。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    tag_matcher = _words_to_matcher(_split_to_words(tag))
    return bool(tag_matcher & blacklist_set)


def is_blacklisted(tag: str) -> bool:
    """タグが既定ブラックリストに含まれるか判定します。

    タグ文字列を受け取り、オンライン由来のブラックリストに一致すれば `True` を返します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    return _is_blacklisted(tag, _online_blacklist_set())


def drop_blacklisted_tags(
    tags: list[str] | Mapping[str, float],
    use_presets: bool = True,
    custom_blacklist: list[str] | None = None,
) -> list[str] | dict[str, float]:
    """ブラックリストに一致するタグを除去します。

    タグリストまたはタグスコア辞書、既定リスト利用フラグ、任意の追加リストを受け取り、
    入力と同じ種類でブラックリスト該当タグを除いた値を返します。

    Parameters
    ----------
    tags : list[str] | Mapping[str, float]
        処理対象のタグリスト、またはタグ名からスコアへのマッピングです。
    use_presets : bool, optional
        既定ブラックリストを使うかを指定します。
    custom_blacklist : list[str] | None
        追加で除外するタグ文字列のリストです。

    Returns
    -------
    returns : list[str] | dict[str, float]
        処理結果を格納した辞書を返します。

    Raises
    ------
    TypeError
        入力値の型が対応していない場合に送出されます。
    """
    blacklist = set()
    if use_presets:
        blacklist = blacklist | _online_blacklist_set()
    for tag in custom_blacklist or []:
        blacklist = blacklist | _words_to_matcher(_split_to_words(tag))

    if isinstance(tags, Mapping):
        return {tag: value for tag, value in tags.items() if not _is_blacklisted(tag, blacklist)}
    if isinstance(tags, list):
        return [tag for tag in tags if not _is_blacklisted(tag, blacklist)]
    raise TypeError(f"Unsupported types of tags, dict or list expected, but {tags!r} found.")
