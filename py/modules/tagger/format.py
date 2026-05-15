import re
from collections.abc import Mapping


RE_SPECIAL = re.compile(r"([\\()])")
_KAOMOJIS = {
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
}


def add_underline(tag: str) -> str:
    """タグ内の空白をアンダースコアへ変換します。

    タグ文字列を受け取り、前後空白を除いたうえで空白を `_` に置換した文字列を返します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。

    Returns
    -------
    returns : str
        変換または生成した文字列を返します。
    """
    return tag.strip().replace(" ", "_")


def remove_underline(tag: str) -> str:
    """タグ内のアンダースコアを空白へ変換します。

    タグ文字列を受け取り、顔文字タグは保持し、それ以外は `_` を空白に置換した文字列を返します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。

    Returns
    -------
    returns : str
        変換または生成した文字列を返します。
    """
    tag = tag.strip()
    return tag.replace("_", " ") if tag not in _KAOMOJIS else tag


def escape_tag(tag: str) -> str:
    """プロンプト用に特殊文字をエスケープします。

    タグ文字列を受け取り、バックスラッシュと丸括弧をエスケープした文字列を返します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。

    Returns
    -------
    returns : str
        変換または生成した文字列を返します。
    """
    return re.sub(RE_SPECIAL, r"\\\1", tag)


def unescape_tag(tag: str) -> str:
    """エスケープ済みの丸括弧とバックスラッシュを元の文字へ戻します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。

    Returns
    -------
    str
        エスケープを解除したタグ文字列です。
    """
    tag = tag.strip()
    tag = tag.replace(r"\(", "(").replace(r"\)", ")")
    tag = tag.replace(r"\\", "\\")
    return tag


def tags_to_text(
    tags: Mapping[str, float],
    use_spaces: bool = False,
    use_escape: bool = True,
    include_score: bool = False,
    score_descend: bool = True,
    trailing_comma: bool = False,
) -> str:
    """タグスコア辞書をカンマ区切りテキストへ変換します。

    タグとスコアの辞書、空白表記・エスケープ・スコア付与・並び替え設定を受け取り、プロンプト保存向け文字列を返します。

    Parameters
    ----------
    tags : Mapping[str, float]
        処理対象のタグリスト、またはタグ名からスコアへのマッピングです。
    use_spaces : bool, optional
        タグ内のアンダースコアを空白に変換するかを指定します。
    use_escape : bool, optional
        特殊文字をエスケープするかを指定します。
    include_score : bool, optional
        タグにスコア表記を付けるかを指定します。
    score_descend : bool, optional
        スコア降順で並び替えるかを指定します。
    trailing_comma : bool, optional
        出力文字列の末尾にカンマを付けるかを指定します。

    Returns
    -------
    returns : str
        変換または生成した文字列を返します。
    """
    tag_pairs = tags.items()
    if score_descend:
        tag_pairs = sorted(tag_pairs, key=lambda x: (-x[1], x[0]))

    text_items = []
    for tag, score in tag_pairs:
        tag_text = remove_underline(tag) if use_spaces else tag
        if use_escape:
            tag_text = escape_tag(tag_text)
        if include_score:
            tag_text = f"({tag_text}:{score:.3f})"
        text_items.append(tag_text)

    text = ", ".join(text_items)
    if trailing_comma and text:
        text += ", "
    return text
