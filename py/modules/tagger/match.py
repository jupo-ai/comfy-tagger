import re
from functools import lru_cache


@lru_cache(maxsize=2048)
def _cached_singular_form(word: str) -> str:
    """英単語の簡易単数形を返します。

    小文字の単語を受け取り、末尾規則に基づく単数形候補を返します。

    Parameters
    ----------
    word : str
        単数形または複数形へ変換する単語です。

    Returns
    -------
    returns : str
        変換または生成した文字列を返します。
    """
    if word.endswith("ies") and len(word) > 3:
        return f"{word[:-3]}y"
    if word.endswith(("ses", "xes", "zes", "ches", "shes")) and len(word) > 2:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 1:
        return word[:-1]
    return word


@lru_cache(maxsize=2048)
def _cached_plural_form(word: str) -> str:
    """英単語の簡易複数形を返します。

    小文字の単語を受け取り、末尾規則に基づく複数形候補を返します。

    Parameters
    ----------
    word : str
        単数形または複数形へ変換する単語です。

    Returns
    -------
    returns : str
        変換または生成した文字列を返します。
    """
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return f"{word[:-1]}ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return f"{word}es"
    return f"{word}s"


def _split_to_words(text: str) -> list[str]:
    """タグ文字列を照合用の単語列へ分割します。

    空白または `_` を含む文字列を受け取り、小文字化した単語リストを返します。

    Parameters
    ----------
    text : str
        照合または変換対象の文字列です。

    Returns
    -------
    returns : list[str]
        処理結果のリストを返します。
    """
    return [word.lower() for word in re.split(r"[\s_]+", text) if word]


def _words_to_matcher(words: list[str], enable_forms: bool = True) -> set[tuple[str, ...]]:
    """単語列から照合候補セットを作ります。

    単語リストと単複数形展開フラグを受け取り、完全一致比較用のタプル集合を返します。

    Parameters
    ----------
    words : list[str]
        照合候補を作る単語リストです。
    enable_forms : bool, optional
        単複数形の候補を含めるかを指定します。

    Returns
    -------
    returns : set[tuple[str, ...]]
        複数の処理結果をまとめたタプルを返します。
    """
    if words and enable_forms:
        words_tuples = [
            words,
            [*words[:-1], _cached_singular_form(words[-1])],
            [*words[:-1], _cached_plural_form(words[-1])],
        ]
    else:
        words_tuples = [words]
    return {tuple(item) for item in words_tuples}


def tag_match_suffix(text: str, suffix: str) -> bool:
    """タグ文字列が指定サフィックスに一致するか判定します。

    タグ文字列とサフィックス文字列を受け取り、末尾単語列が単複数形を含めて一致すれば `True` を返します。

    Parameters
    ----------
    text : str
        照合または変換対象の文字列です。
    suffix : str
        末尾一致で比較するサフィックス文字列です。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    suffix_words = _split_to_words(suffix)
    text_words = _split_to_words(text)
    if not suffix_words:
        return True

    text_words = text_words[-len(suffix_words):]
    return bool(_words_to_matcher(text_words) & _words_to_matcher(suffix_words))


def tag_match_prefix(text: str, prefix: str) -> bool:
    """タグ文字列が指定プレフィックスに一致するか判定します。

    タグ文字列とプレフィックス文字列を受け取り、先頭単語列が一致すれば `True` を返します。

    Parameters
    ----------
    text : str
        照合または変換対象の文字列です。
    prefix : str
        先頭一致で比較するプレフィックス文字列です。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    prefix_words = _split_to_words(prefix)
    text_words = _split_to_words(text)
    if not prefix_words:
        return True

    text_words = text_words[:len(prefix_words)]
    return bool(
        _words_to_matcher(text_words, enable_forms=False)
        & _words_to_matcher(prefix_words, enable_forms=False)
    )


def tag_match_full(t1: str, t2: str) -> bool:
    """2つのタグ文字列が単語列として完全一致するか判定します。

    比較する2つのタグ文字列を受け取り、単複数形を含めて一致すれば `True` を返します。

    Parameters
    ----------
    t1 : str
        比較対象の1つ目のタグ文字列です。
    t2 : str
        比較対象の2つ目のタグ文字列です。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    t1_words = _split_to_words(t1)
    t2_words = _split_to_words(t2)
    return bool(_words_to_matcher(t1_words) & _words_to_matcher(t2_words))
