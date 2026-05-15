from collections.abc import Mapping

from .match import _split_to_words, _words_to_matcher


CHAR_WHITELIST_SUFFIX = [
    "anal_hair",
    "anal_tail",
    "arm_behind_head",
    "arm_hair",
    "arm_under_breasts",
    "arms_behind_head",
    "bird_on_head",
    "blood_in_hair",
    "breasts_on_glass",
    "breasts_on_head",
    "cat_on_head",
    "closed_eyes",
    "clothed_female_nude_female",
    "clothed_female_nude_male",
    "clothed_male_nude_female",
    "clothes_between_breasts",
    "cream_on_face",
    "drying_hair",
    "empty_eyes",
    "face_to_breasts",
    "facial",
    "food_on_face",
    "food_on_head",
    "game_boy",
    "grabbing_another's_hair",
    "grabbing_own_breast",
    "gun_to_head",
    "half-closed_eyes",
    "head_between_breasts",
    "heart_in_eye",
    "multiple_boys",
    "multiple_girls",
    "object_on_breast",
    "object_on_head",
    "paint_splatter_on_face",
    "parted_lips",
    "penis_on_face",
    "person_on_head",
    "pokemon_on_head",
    "pubic_hair",
    "rabbit_on_head",
    "rice_on_face",
    "severed_head",
    "star_in_eye",
    "sticker_on_face",
    "tentacles_on_male",
    "tying_hair",
]
CHAR_WHITELIST_PREFIX = [
    "holding",
    "hand on",
    "hands on",
    "hand to",
    "hands to",
    "hand in",
    "hands in",
    "hand over",
    "hands over",
    "futa with",
    "futa on",
    "cum on",
    "covering",
    "adjusting",
    "rubbing",
    "sitting",
    "shading",
    "playing",
    "cutting",
]
CHAR_WHITELIST_WORD = ["drill"]
CHAR_SUFFIXES = [
    "eyes",
    "skin",
    "hair",
    "bun",
    "bangs",
    "cut",
    "sidelocks",
    "twintails",
    "braid",
    "braids",
    "afro",
    "ahoge",
    "drill",
    "drills",
    "bald",
    "dreadlocks",
    "side up",
    "ponytail",
    "updo",
    "beard",
    "mustache",
    "pointy ears",
    "ear",
    "horn",
    "tail",
    "wing",
    "ornament",
    "hairband",
    "pupil",
    "bow",
    "eyewear",
    "headwear",
    "ribbon",
    "crown",
    "cap",
    "hat",
    "hairclip",
    "breast",
    "mole",
    "halo",
    "earrings",
    "animal ear fluff",
    "hair flower",
    "glasses",
    "fang",
    "female",
    "girl",
    "boy",
    "male",
    "beret",
    "heterochromia",
    "headdress",
    "headgear",
    "eyepatch",
    "headphones",
    "eyebrows",
    "eyelashes",
    "sunglasses",
    "hair intakes",
    "scrunchie",
    "ear_piercing",
    "head",
    "on face",
    "on head",
    "on hair",
    "headband",
    "hair rings",
    "under_mouth",
    "freckles",
    "lip",
    "eyeliner",
    "eyeshadow",
    "tassel",
    "over one eye",
    "drill",
    "drill hair",
]
CHAR_PREFIXES = ["hair over", "hair between", "facial"]


class _WordPool:
    """完全一致で照合する単語列プールです。

    単語候補リストを受け取り、`in` 演算で文字列が登録済み候補に一致するか判定できる内部用クラスです。

    Parameters
    ----------
    words : list[str] | None
        照合候補を作る単語リストです。
    """

    def __init__(self, words: list[str] | None = None) -> None:
        """単語候補を照合用データへ変換して保持します。

        Parameters
        ----------
        words : list[str] | None
            照合候補を作る単語リストです。

        Returns
        -------
        returns : None
            値は返しません。
        """
        self._words: dict[int, set[tuple[str, ...]]] = {}
        for word in words or []:
            self._append(word)

    def _append(self, text: str) -> None:
        """文字列を単語列候補へ変換し、プールへ追加します。

        Parameters
        ----------
        text : str
            照合または変換対象の文字列です。

        Returns
        -------
        returns : None
            値は返しません。
        """
        for item in _words_to_matcher(_split_to_words(text)):
            self._words.setdefault(len(item), set()).add(item)

    def __contains__(self, text: str) -> bool:
        """文字列がプール内の単語列と完全一致するかを返します。

        Parameters
        ----------
        text : str
            照合または変換対象の文字列です。

        Returns
        -------
        returns : bool
            登録済み候補に完全一致する場合は True を返します。
        """
        words = tuple(_split_to_words(text))
        return len(words) in self._words and words in self._words[len(words)]


class _SuffixPool:
    """末尾一致で照合する単語列プールです。

    サフィックス候補リストを受け取り、`in` 演算で文字列末尾が候補に一致するか判定できる内部用クラスです。

    Parameters
    ----------
    suffixes : list[str] | None
        基本キャラクタータグとみなすサフィックス候補です。
    """

    def __init__(self, suffixes: list[str] | None = None) -> None:
        """サフィックス候補を照合用データへ変換して保持します。

        Parameters
        ----------
        suffixes : list[str] | None
            基本キャラクタータグとみなすサフィックス候補です。

        Returns
        -------
        returns : None
            値は返しません。
        """
        self._suffixes: dict[int, set[tuple[str, ...]]] = {}
        for suffix in suffixes or []:
            self._append(suffix)

    def _append(self, text: str) -> None:
        """サフィックス文字列を単語列候補へ変換し、プールへ追加します。

        Parameters
        ----------
        text : str
            照合または変換対象の文字列です。

        Returns
        -------
        returns : None
            値は返しません。
        """
        for item in _words_to_matcher(_split_to_words(text)):
            self._suffixes.setdefault(len(item), set()).add(item)

    def __contains__(self, text: str) -> bool:
        """文字列末尾がプール内のサフィックス候補に一致するかを返します。

        Parameters
        ----------
        text : str
            照合または変換対象の文字列です。

        Returns
        -------
        returns : bool
            登録済みサフィックスに一致する場合は True を返します。
        """
        words = _split_to_words(text)
        for length, tpl_set in self._suffixes.items():
            if length > len(words):
                continue

            seg = [] if length == 0 else words[-length:]
            if _words_to_matcher(seg) & tpl_set:
                return True
        return False


class _PrefixPool:
    """先頭一致で照合する単語列プールです。

    プレフィックス候補リストを受け取り、`in` 演算で文字列先頭が候補に一致するか判定できる内部用クラスです。

    Parameters
    ----------
    prefixes : list[str] | None
        基本キャラクタータグとみなすプレフィックス候補です。
    """

    def __init__(self, prefixes: list[str] | None = None) -> None:
        """プレフィックス候補を照合用データへ変換して保持します。

        Parameters
        ----------
        prefixes : list[str] | None
            基本キャラクタータグとみなすプレフィックス候補です。

        Returns
        -------
        returns : None
            値は返しません。
        """
        self._prefixes: dict[int, set[tuple[str, ...]]] = {}
        for prefix in prefixes or []:
            self._append(prefix)

    def _append(self, text: str) -> None:
        """プレフィックス文字列を単語列候補へ変換し、プールへ追加します。

        Parameters
        ----------
        text : str
            照合または変換対象の文字列です。

        Returns
        -------
        returns : None
            値は返しません。
        """
        for item in _words_to_matcher(_split_to_words(text), enable_forms=False):
            self._prefixes.setdefault(len(item), set()).add(item)

    def __contains__(self, text: str) -> bool:
        """文字列先頭がプール内のプレフィックス候補に一致するかを返します。

        Parameters
        ----------
        text : str
            照合または変換対象の文字列です。

        Returns
        -------
        returns : bool
            登録済みプレフィックスに一致する場合は True を返します。
        """
        words = _split_to_words(text)
        for length, tpl_set in self._prefixes.items():
            if length > len(words):
                continue

            seg = words[:length]
            if _words_to_matcher(seg, enable_forms=False) & tpl_set:
                return True
        return False


class CharacterTagPool:
    """基本的なキャラクター外観タグを判定するプールです。

    ホワイトリストと基本キャラクタータグ候補を受け取り、タグを除去対象の基本キャラタグか判定します。

    Parameters
    ----------
    whitelist_suffixes : list[str] | None
        除去対象から外すサフィックス候補です。
    whitelist_prefixes : list[str] | None
        除去対象から外すプレフィックス候補です。
    whitelist_words : list[str] | None
        除去対象から外す完全一致単語候補です。
    suffixes : list[str] | None
        基本キャラクタータグとみなすサフィックス候補です。
    prefixes : list[str] | None
        基本キャラクタータグとみなすプレフィックス候補です。

    Raises
    ------
    TypeError
        入力値の型が対応していない場合に送出されます。
    """

    def __init__(
        self,
        whitelist_suffixes: list[str] | None = None,
        whitelist_prefixes: list[str] | None = None,
        whitelist_words: list[str] | None = None,
        suffixes: list[str] | None = None,
        prefixes: list[str] | None = None,
    ) -> None:
        """判定用のホワイトリストと基本タグ候補を初期化します。

        Parameters
        ----------
        whitelist_suffixes : list[str] | None
            除去対象から外すサフィックス候補です。
        whitelist_prefixes : list[str] | None
            除去対象から外すプレフィックス候補です。
        whitelist_words : list[str] | None
            除去対象から外す完全一致単語候補です。
        suffixes : list[str] | None
            基本キャラクタータグとみなすサフィックス候補です。
        prefixes : list[str] | None
            基本キャラクタータグとみなすプレフィックス候補です。

        Returns
        -------
        returns : None
            値は返しません。
        """
        self._whitelist_suffix = _SuffixPool(whitelist_suffixes or CHAR_WHITELIST_SUFFIX)
        self._whitelist_prefix = _PrefixPool(whitelist_prefixes or CHAR_WHITELIST_PREFIX)
        self._whitelist_words = _WordPool(whitelist_words or CHAR_WHITELIST_WORD)
        self._suffixes = _SuffixPool(suffixes or CHAR_SUFFIXES)
        self._prefixes = _PrefixPool(prefixes or CHAR_PREFIXES)

    def _is_in_whitelist(self, tag: str) -> bool:
        """タグが除去しない例外リストに含まれるかを返します。

        Parameters
        ----------
        tag : str
            対象のタグ文字列です。

        Returns
        -------
        returns : bool
            条件を満たす場合は True、満たさない場合は False を返します。
        """
        return (tag in self._whitelist_words) or (tag in self._whitelist_suffix) or (tag in self._whitelist_prefix)

    def _is_in_common(self, tag: str) -> bool:
        """タグが基本キャラクタータグ候補に含まれるかを返します。

        Parameters
        ----------
        tag : str
            対象のタグ文字列です。

        Returns
        -------
        returns : bool
            条件を満たす場合は True、満たさない場合は False を返します。
        """
        return (tag in self._suffixes) or (tag in self._prefixes)

    def is_basic_character_tag(self, tag: str) -> bool:
        """タグが基本キャラクター外観タグか判定します。

        タグ文字列を受け取り、ホワイトリストに含まれず基本候補に一致する場合 `True` を返します。

        Parameters
        ----------
        tag : str
            対象のタグ文字列です。

        Returns
        -------
        returns : bool
            条件を満たす場合は True、満たさない場合は False を返します。
        """
        if self._is_in_whitelist(tag):
            return False
        return self._is_in_common(tag)

    def drop_basic_character_tags(self, tags: list[str] | Mapping[str, float]) -> list[str] | dict[str, float]:
        """基本キャラクター外観タグを除去します。

        タグリストまたはタグスコア辞書を受け取り、入力と同じ種類で基本キャラタグを除いた値を返します。

        Parameters
        ----------
        tags : list[str] | Mapping[str, float]
            処理対象のタグリスト、またはタグ名からスコアへのマッピングです。

        Returns
        -------
        returns : list[str] | dict[str, float]
            処理結果を格納した辞書を返します。

        Raises
        ------
        TypeError
            入力値の型が対応していない場合に送出されます。
        """
        if isinstance(tags, Mapping):
            return {tag: value for tag, value in tags.items() if not self.is_basic_character_tag(tag)}
        if isinstance(tags, list):
            return [tag for tag in tags if not self.is_basic_character_tag(tag)]
        raise TypeError(f"Unsupported types of tags, dict or list expected, but {tags!r} found.")


_DEFAULT_CHARACTER_POOL = CharacterTagPool()


def is_basic_character_tag(tag: str) -> bool:
    """既定プールで基本キャラクター外観タグか判定します。

    タグ文字列を受け取り、基本キャラタグなら `True` を返します。

    Parameters
    ----------
    tag : str
        対象のタグ文字列です。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    return _DEFAULT_CHARACTER_POOL.is_basic_character_tag(tag)


def drop_basic_character_tags(tags: list[str] | Mapping[str, float]) -> list[str] | dict[str, float]:
    """既定プールで基本キャラクター外観タグを除去します。

    タグリストまたはタグスコア辞書を受け取り、入力と同じ種類で除去後の値を返します。

    Parameters
    ----------
    tags : list[str] | Mapping[str, float]
        処理対象のタグリスト、またはタグ名からスコアへのマッピングです。

    Returns
    -------
    returns : list[str] | dict[str, float]
        処理結果を格納した辞書を返します。
    """
    return _DEFAULT_CHARACTER_POOL.drop_basic_character_tags(tags)
