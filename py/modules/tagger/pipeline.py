from pathlib import Path

import torch

from .order import SortMode
from .postprocess import TagPostprocessOptions, categorize_tags, ordered_tags_from_groups, tags_to_prompt_text
from .registry import get_model_spec
from .runtime import image_to_uint8, open_onnx_session
from .wd14 import predict_wd14_tags
from .pixai import predict_pixai_tags
from .oppai_oracle import predict_oppai_oracle_tags


def _raw_output_for_image(
    image_np,
    spec,
    session,
    model_paths: dict[str, Path],
) -> dict:
    if spec.family == "wd14":
        return predict_wd14_tags(
            image_np,
            session,
            csv_path=model_paths["selected_tags.csv"],
        )
    if spec.family == "pixai":
        return predict_pixai_tags(
            image_np,
            session,
            csv_path=model_paths["selected_tags.csv"],
            thresholds_path=model_paths.get("thresholds.csv"),
            preprocess_path=model_paths.get("preprocess.json"),
        )
    if spec.family == "oppai_oracle":
        return predict_oppai_oracle_tags(
            image_np,
            session,
            csv_path=model_paths["selected_tags.csv"],
            preprocess_path=model_paths.get("preprocessing.json"),
        )
    raise ValueError(f"Unsupported tagger family: {spec.family}")


def tag_image_grouped(
    image: torch.Tensor,
    model: str,
    model_paths: dict[str, Path],
    threshold: float,
    include_rating: bool = False,
    rating_threshold: float = 0.35,
    include_character: bool = True,
    character_threshold: float = 0.85,
    include_copyright: bool = False,
    copyright_threshold: float = 0.35,
    include_meta: bool = False,
    meta_threshold: float = 0.35,
    replace_underscore: bool = True,
    exclude_tags: str = "",
    drop_overlap: bool = False,
    drop_blacklist: bool = False,
    drop_basic_character: bool = False,
    sort_mode: SortMode = "original",
    prioritize_people_tags: bool = True,
) -> dict[str, dict[str, float]]:
    spec = get_model_spec(model)
    session = open_onnx_session(model_paths["model.onnx"])
    image_np = image_to_uint8(image)
    output = _raw_output_for_image(image_np, spec, session, model_paths)

    return categorize_tags(
        output,
        TagPostprocessOptions(
            replace_underscore=replace_underscore,
            exclude_tags=exclude_tags,
            drop_overlap=drop_overlap,
            drop_blacklist=drop_blacklist,
            drop_basic_character=drop_basic_character,
            sort_mode=sort_mode,
            prioritize_people_tags=prioritize_people_tags,
            threshold=threshold,
            include_rating=include_rating,
            rating_threshold=rating_threshold,
            include_character=include_character,
            character_threshold=character_threshold,
            include_copyright=include_copyright,
            copyright_threshold=copyright_threshold,
            include_meta=include_meta,
            meta_threshold=meta_threshold,
        ),
    )


def tag_image(
    image: torch.Tensor,
    model: str,
    model_paths: dict[str, Path],
    threshold: float,
    include_rating: bool = False,
    rating_threshold: float = 0.35,
    include_character: bool = True,
    character_threshold: float = 0.85,
    include_copyright: bool = False,
    copyright_threshold: float = 0.35,
    include_meta: bool = False,
    meta_threshold: float = 0.35,
    replace_underscore: bool = True,
    trailing_comma: bool = False,
    exclude_tags: str = "",
    drop_overlap: bool = False,
    drop_blacklist: bool = False,
    drop_basic_character: bool = False,
    sort_mode: SortMode = "original",
    prioritize_people_tags: bool = True,
) -> str:
    grouped = tag_image_grouped(
        image=image,
        model=model,
        model_paths=model_paths,
        threshold=threshold,
        include_rating=include_rating,
        rating_threshold=rating_threshold,
        include_character=include_character,
        character_threshold=character_threshold,
        include_copyright=include_copyright,
        copyright_threshold=copyright_threshold,
        include_meta=include_meta,
        meta_threshold=meta_threshold,
        replace_underscore=replace_underscore,
        exclude_tags=exclude_tags,
        drop_overlap=drop_overlap,
        drop_blacklist=drop_blacklist,
        drop_basic_character=drop_basic_character,
        sort_mode=sort_mode,
        prioritize_people_tags=prioritize_people_tags,
    )
    tags = ordered_tags_from_groups(grouped)
    return tags_to_prompt_text(tags, trailing_comma=trailing_comma)
