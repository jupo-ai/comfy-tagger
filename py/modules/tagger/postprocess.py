from dataclasses import dataclass

from .blacklist import drop_blacklisted_tags
from .character import drop_basic_character_tags
from .format import remove_underline, tags_to_text
from .order import SortMode, sort_tags
from .overlap import drop_overlap_tags
from .tag_catalog import is_character_tag, is_copyright_tag, is_meta_tag, is_rating_tag


@dataclass(frozen=True)
class TagPostprocessOptions:
    threshold: float
    include_rating: bool = False
    rating_threshold: float = 0.35
    include_character: bool = True
    character_threshold: float = 0.85
    include_copyright: bool = False
    copyright_threshold: float = 0.35
    include_meta: bool = False
    meta_threshold: float = 0.35
    replace_underscore: bool = True
    exclude_tags: str = ""
    drop_overlap: bool = False
    drop_blacklist: bool = False
    drop_basic_character: bool = False
    sort_mode: SortMode = "original"
    prioritize_people_tags: bool = True
    trailing_comma: bool = False


def _normalize_tag(tag: str, replace_underscore: bool) -> str:
    return remove_underline(tag) if replace_underscore else tag


def _normalize_tag_mapping(tags: object, replace_underscore: bool) -> dict[str, float]:
    if not isinstance(tags, dict):
        return {}
    return {_normalize_tag(tag, replace_underscore): float(score) for tag, score in tags.items()}


def _merge_tag_mapping(target: dict[str, float], source: object, replace_underscore: bool) -> None:
    for tag, score in _normalize_tag_mapping(source, replace_underscore).items():
        target.setdefault(tag, score)


def _category_for_tag(tag: str, explicit_categories: dict[str, str]) -> str:
    if tag in explicit_categories:
        return explicit_categories[tag]
    if is_rating_tag(tag):
        return "rating"
    if is_character_tag(tag):
        return "character"
    if is_copyright_tag(tag):
        return "copyright"
    if is_meta_tag(tag):
        return "meta"
    return "general"


def _exclude_tags(tags: dict[str, float], exclude_tags: str) -> dict[str, float]:
    exclude = {tag.strip().lower() for tag in exclude_tags.split(",") if tag.strip()}
    if not exclude:
        return tags
    return {tag: score for tag, score in tags.items() if tag.lower() not in exclude}


def _filter_tags(
    tags: dict[str, float],
    exclude_tags: str,
    drop_blacklist: bool,
    drop_basic_character: bool,
) -> dict[str, float]:
    tags = _exclude_tags(tags, exclude_tags)
    if drop_blacklist:
        tags = drop_blacklisted_tags(tags)
    if drop_basic_character:
        tags = drop_basic_character_tags(tags)
    return tags


def _sort_tag_mapping(
    tags: dict[str, float],
    sort_mode: SortMode,
    prioritize_people_tags: bool,
) -> dict[str, float]:
    return {tag: tags[tag] for tag in sort_tags(tags, mode=sort_mode, prioritize_people_tags=prioritize_people_tags)}


def categorize_tags(output: dict, options: TagPostprocessOptions) -> dict[str, dict[str, float]]:
    all_tags: dict[str, float] = {}
    for group_name in ("rating", "character", "copyright", "meta", "general", "tag"):
        _merge_tag_mapping(all_tags, output.get(group_name), options.replace_underscore)

    explicit_categories: dict[str, str] = {}
    for group_name in ("rating", "character", "copyright", "meta"):
        for tag in _normalize_tag_mapping(output.get(group_name), options.replace_underscore):
            explicit_categories[tag] = group_name

    grouped = {
        "rating": {},
        "character": {},
        "copyright": {},
        "meta": {},
        "general": {},
    }
    thresholds = {
        "rating": options.rating_threshold,
        "character": options.character_threshold,
        "copyright": options.copyright_threshold,
        "meta": options.meta_threshold,
        "general": options.threshold,
    }
    includes = {
        "rating": options.include_rating,
        "character": options.include_character,
        "copyright": options.include_copyright,
        "meta": options.include_meta,
        "general": True,
    }

    for tag, score in all_tags.items():
        category = _category_for_tag(tag, explicit_categories)
        if not includes[category] or score <= thresholds[category]:
            continue
        grouped[category][tag] = score

    if options.drop_overlap:
        grouped["general"] = drop_overlap_tags(grouped["general"])

    for group_name, group_tags in grouped.items():
        group_tags = _filter_tags(
            group_tags,
            exclude_tags=options.exclude_tags,
            drop_blacklist=options.drop_blacklist,
            drop_basic_character=options.drop_basic_character,
        )
        grouped[group_name] = _sort_tag_mapping(
            group_tags,
            sort_mode=options.sort_mode,
            prioritize_people_tags=options.prioritize_people_tags,
        )

    return grouped


def ordered_tags_from_groups(grouped: dict[str, dict[str, float]]) -> dict[str, float]:
    ordered_tags: dict[str, float] = {}
    for group_name in ("character", "copyright", "general", "meta", "rating"):
        ordered_tags.update(grouped.get(group_name, {}))
    return ordered_tags


def tags_to_prompt_text(tags: dict[str, float], trailing_comma: bool = False) -> str:
    return tags_to_text(
        tags,
        use_escape=True,
        include_score=False,
        score_descend=False,
        trailing_comma=trailing_comma,
    )
