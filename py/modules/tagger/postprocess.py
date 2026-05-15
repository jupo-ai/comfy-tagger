import numpy as np
from dataclasses import dataclass

from .blacklist import drop_blacklisted_tags
from .overlap import drop_overlap_tags
from .character import drop_basic_character_tags
from .format import add_underline, remove_underline, escape_tag, unescape_tag, tags_to_text
from .order import SortMode, sort_tags
from .tag_catalog import is_artist_tag, is_character_tag, is_copyright_tag, is_meta_tag, is_rating_tag


@dataclass(frozen=True)
class TagPostprocessOptions:
    """タグ推論結果へ適用する共通後処理オプションです。

    カテゴリごとのしきい値、出力対象カテゴリ、表記変換、除外、重複除去、並び替え設定を受け取り、
    `execute_postprocess()` の振る舞いを決める不変の設定値として使われます。

    Parameters
    ----------
    threshold : float
        一般タグに適用するしきい値です。
    include_general : bool
        一般タグを出力に含めるかを示します。
    include_rating : bool
        レーティングタグを出力に含めるかを示します。
    rating_threshold : float
        レーティングタグのしきい値です。
    include_character : bool
        キャラクタータグを出力に含めるかを示します。
    character_threshold : float
        キャラクタータグのしきい値です。
    include_artist : bool
        アーティストタグを出力に含めるかを示します。
    artist_threshold : float
        アーティストタグのしきい値です。
    include_copyright : bool
        版権タグを出力に含めるかを示します。
    copyright_threshold : float
        版権タグのしきい値です。
    include_meta : bool
        メタタグを出力に含めるかを示します。
    meta_threshold : float
        メタタグのしきい値です。
    mcut_threshold : bool
        各thresholdをmcut法で自動計算するかを示します。
    replace_underscore : bool
        タグ表記のアンダースコアを空白に変換するかを示します。
    escape_brackets : bool
        括弧 `()` をエスケープするかを示します。
    exclude_tags : str
        除外タグのカンマ区切り文字列です。
    drop_overlap : bool
        重複タグを除去するかを示します。
    drop_blacklist : bool
        ブラックリストタグを除去するかを示します。
    drop_basic_character : bool
        基本キャラクタータグを除去するかを示します。
    sort_mode : SortMode
        タグの並び替えモードです。
    prioritize_people_tags : bool
        人数タグを優先するかを示します。
    trailing_comma : bool
        末尾カンマを付与するかを示します。
    """

    threshold: float = 0.35
    include_general: bool = True
    include_rating: bool = False
    rating_threshold: float = 0.35
    include_character: bool = True
    character_threshold: float = 0.85
    include_artist: bool = False
    artist_threshold: float = 0.35
    include_copyright: bool = False
    copyright_threshold: float = 0.35
    include_meta: bool = False
    meta_threshold: float = 0.35
    mcut_threshold: bool = False
    replace_underscore: bool = True
    escape_brackets: bool = True
    exclude_tags: str = ""
    drop_overlap: bool = False
    drop_blacklist: bool = False
    drop_basic_character: bool = False
    sort_mode: SortMode = "original"
    prioritize_people_tags: bool = True
    trailing_comma: bool = False


def _normalize_tag(tag: str) -> str:
    """タグを内部照合用のアンダースコア表記へ正規化します。

    Parameters
    ----------
    tag : str
        入力タグです。

    Returns
    -------
    str
        エスケープを外し、空白をアンダースコアへ変換したタグです。
    """
    new_tag = unescape_tag(tag)
    new_tag = add_underline(new_tag)
    return new_tag


def _category_for_tag(tag: str) -> str:
    """タグカタログに基づいてタグカテゴリ名を返します。

    Parameters
    ----------
    tag : str
        内部表記のタグです。

    Returns
    -------
    str
        `rating`、`artist`、`character`、`copyright`、`meta`、`general` のいずれかです。
    """
    if is_rating_tag(tag):
        return "rating"
    if is_artist_tag(tag):
        return "artist"
    if is_character_tag(tag):
        return "character"
    if is_copyright_tag(tag):
        return "copyright"
    if is_meta_tag(tag):
        return "meta"
    return "general"


def _mcut_threshold(probs: np.ndarray) -> float:
    """MCut 法で自動しきい値を計算します。

    確率配列を受け取り、隣接スコア差が最大になる境界の中点をしきい値として返します。

    Parameters
    ----------
    probs : np.ndarray
        しきい値計算に使う確率配列です。

    Returns
    -------
    returns : float
        計算した浮動小数点値を返します。
    """
    if len(probs) < 2:
        return 0.0
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    index = int(difs.argmax())
    return float((sorted_probs[index] + sorted_probs[index + 1]) / 2)


def _exclude_tags(tags: dict[str, float], exclude_tags: str) -> dict[str, float]:
    """カンマ区切りの除外指定に一致するタグを取り除きます。

    タグスコア辞書と除外タグ文字列を受け取り、除外指定に一致しないタグだけの辞書を返します。

    Parameters
    ----------
    tags : dict[str, float]
        処理対象のタグリスト、またはタグ名からスコアへのマッピングです。
    exclude_tags : str
        除外するタグをカンマ区切りで指定する文字列です。

    Returns
    -------
    returns : dict[str, float]
        処理結果を格納した辞書を返します。
    """
    exclude = {tag.strip().lower() for tag in exclude_tags.split(",") if tag.strip()}
    exclude = {_normalize_tag(tag) for tag in exclude}
    if not exclude:
        return tags
    return {tag: score for tag, score in tags.items() if tag.lower() not in exclude}


def execute_postprocess(output: dict, options: TagPostprocessOptions) -> dict[str, dict[str, float]]:
    """モデルの生出力へ共通後処理を適用します。

    family ごとの推論関数が返す `{"tag": {tag: score}}` 形式の辞書を受け取り、
    カテゴリ分類、しきい値、除外、ブラックリスト、重複除去、基本キャラクタータグ除去、
    並び替え、表記変換を順に適用します。

    Parameters
    ----------
    output : dict
        推論関数が返した生出力です。
    options : TagPostprocessOptions
        後処理設定です。

    Returns
    -------
    dict[str, dict[str, float]]
        カテゴリ名からタグスコア辞書へのマッピングです。
    """
    all_tags = output.get("tag", {})
    all_tags = {_normalize_tag(t): float(s) for t, s in all_tags.items()}

    grouped = {
        "rating": {},
        "artist": {},
        "character": {},
        "copyright": {},
        "meta": {},
        "general": {},
    }
    includes = {
        "rating": options.include_rating,
        "artist": options.include_artist,
        "character": options.include_character,
        "copyright": options.include_copyright,
        "meta": options.include_meta,
        "general": options.include_general,
    }
    thresholds = {
        "rating": options.rating_threshold, 
        "artist": options.artist_threshold, 
        "character": options.character_threshold, 
        "copyright": options.copyright_threshold, 
        "meta": options.meta_threshold, 
        "general": options.threshold, 
    }
    
    # タグをカテゴライズ
    for tag, score in all_tags.items():
        category = _category_for_tag(tag)
        grouped[category][tag] = score
    
    # カテゴリごとにポストプロセス
    for category in list(grouped.keys()):
        tags = grouped.get(category, {})
        
        # include チェック
        include = includes.get(category, False)
        if not include:
            grouped[category] = {}
            continue
        
        # threshold フィルタ
        threshold = thresholds.get(category, 1.0)
        if options.mcut_threshold:
            probs = np.array(list(tags.values()))
            threshold = max(0.15, _mcut_threshold(probs)) # 念のため最小値を小さめの値に
        tags = {tag: score for tag, score in tags.items() if score >= threshold}
        
        # exclude フィルタ
        tags = _exclude_tags(tags, options.exclude_tags)

        # blacklist フィルタ
        if options.drop_blacklist:
            tags = drop_blacklisted_tags(tags)
        
        # overlap フィルタ
        if options.drop_overlap:
            tags = drop_overlap_tags(tags)
        
        # basic character フィルタ
        if options.drop_basic_character:
            tags = drop_basic_character_tags(tags)
        
        # カテゴリ内の並び替え
        sorted_tag_names = sort_tags(
            tags=tags, 
            mode=options.sort_mode, 
            prioritize_people_tags=options.prioritize_people_tags
        )
        tags = {tag: tags[tag] for tag in sorted_tag_names}
        
        # underscore, escape
        if options.replace_underscore:
            tags = {remove_underline(tag): score for tag, score in tags.items()}
        
        if options.escape_brackets:
            tags = {escape_tag(tag): score for tag, score in tags.items()}
        
        grouped[category] = tags
    
    return grouped



def ordered_tags_from_groups(grouped: dict[str, dict[str, float]]) -> dict[str, float]:
    """カテゴリ別タグ辞書をプロンプト用の固定順で平坦化します。

    Parameters
    ----------
    grouped : dict[str, dict[str, float]]
        `execute_postprocess()` が返すカテゴリ別タグスコア辞書です。

    Returns
    -------
    dict[str, float]
        `artist/character/copyright/general/meta/rating` 順に結合したタグスコア辞書です。
    """
    ordered_tags: dict[str, float] = {}
    for group_name in ("artist", "character", "copyright", "general", "meta", "rating"):
        ordered_tags.update(grouped.get(group_name, {}))
    return ordered_tags


def tags_to_prompt_text(tags: dict[str, float], trailing_comma: bool = False) -> str:
    """タグスコア辞書をプロンプト保存用テキストへ変換します。

    Parameters
    ----------
    tags : dict[str, float]
        出力対象のタグスコア辞書です。
    trailing_comma : bool, optional
        末尾にカンマと空白を付けるかを指定します。

    Returns
    -------
    str
        カンマ区切りのプロンプト文字列です。
    """
    return tags_to_text(
        tags,
        use_escape=True,
        include_score=False,
        score_descend=False,
        trailing_comma=trailing_comma,
    )


