import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .format import remove_underline
from .overlap import drop_overlap_tags


_LABEL_CACHE: dict[tuple[str, bool], tuple[list[str], dict[str, list[int]]]] = {}
_PREPROCESS_CACHE: dict[str, dict] = {}


def load_oppai_oracle_labels(csv_path: Path, no_underline: bool = False) -> tuple[list[str], dict[str, list[int]]]:
    key = (str(csv_path), no_underline)
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]

    tag_names: list[str] = []
    indexes_by_group: dict[str, list[int]] = {"general": []}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
        if records and "id" in records[0]:
            records.sort(key=lambda item: int(item.get("id") or 0))

        for index, row in enumerate(records):
            tag = row.get("name") or row.get("tag") or row.get("tag_name") or ""
            if no_underline:
                tag = remove_underline(tag)
            tag_names.append(tag)

            category = int(row.get("category", 0) or 0) if str(row.get("category", "")).strip() else 0
            if category == 9:
                group = "rating"
            elif category == 4:
                group = "character"
            else:
                group = "general"
            indexes_by_group.setdefault(group, []).append(index)

    value = (tag_names, indexes_by_group)
    _LABEL_CACHE[key] = value
    return value


def load_oppai_oracle_preprocess(preprocess_path: Path | None) -> dict:
    if preprocess_path is None or not preprocess_path.exists():
        return {
            "image_size": 448,
            "pad_color_rgb": [114, 114, 114],
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5],
        }

    key = str(preprocess_path)
    if key in _PREPROCESS_CACHE:
        return _PREPROCESS_CACHE[key]

    try:
        data = json.loads(preprocess_path.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    data.setdefault("image_size", 448)
    data.setdefault("pad_color_rgb", [114, 114, 114])
    data.setdefault("normalize_mean", [0.5, 0.5, 0.5])
    data.setdefault("normalize_std", [0.5, 0.5, 0.5])
    _PREPROCESS_CACHE[key] = data
    return data


def _target_size_from_session(session, preprocess: dict) -> int:
    for input_info in session.get_inputs():
        if input_info.name == "pixel_values":
            int_dims = [dim for dim in input_info.shape if isinstance(dim, int)]
            if len(int_dims) >= 2:
                return int(max(int_dims[-2:]))
    return int(preprocess.get("image_size", 448))


def prepare_image_for_oppai_oracle(
    image_np: np.ndarray,
    target_size: int,
    preprocess: dict,
) -> tuple[np.ndarray, np.ndarray]:
    image = Image.fromarray(image_np, mode="RGB")
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(resized_size, Image.Resampling.BICUBIC)

    pad_color = tuple(int(value) for value in preprocess.get("pad_color_rgb", [114, 114, 114]))
    padded = Image.new("RGB", (target_size, target_size), pad_color)
    offset = ((target_size - resized_size[0]) // 2, (target_size - resized_size[1]) // 2)
    padded.paste(resized, offset)

    padding_mask = np.ones((target_size, target_size), dtype=np.bool_)
    x0, y0 = offset
    padding_mask[y0 : y0 + resized_size[1], x0 : x0 + resized_size[0]] = False

    array = np.asarray(padded, dtype=np.float32) / 255.0
    mean = np.asarray(preprocess.get("normalize_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.asarray(preprocess.get("normalize_std", [0.5, 0.5, 0.5]), dtype=np.float32)
    array = (array - mean) / std
    array = array.transpose(2, 0, 1)
    return np.expand_dims(array, axis=0).astype(np.float32), np.expand_dims(padding_mask, axis=0)


def postprocess_oppai_oracle(
    pred: np.ndarray,
    csv_path: Path,
    threshold: float = 0.753,
    no_underline: bool = False,
    drop_overlap: bool = False,
) -> dict[str, dict[str, float] | np.ndarray]:
    tag_names, indexes_by_group = load_oppai_oracle_labels(csv_path, no_underline)
    result: dict[str, dict[str, float] | np.ndarray] = {}
    all_tags: dict[str, float] = {}

    for group_name, indexes in indexes_by_group.items():
        selected = {
            tag_names[index]: float(pred[index])
            for index in indexes
            if index < len(pred)
            and tag_names[index]
            and tag_names[index] not in {"<PAD>", "<UNK>"}
            and float(pred[index]) > threshold
        }
        if drop_overlap and group_name == "general":
            selected = drop_overlap_tags(selected)
        result[group_name] = selected
        if group_name != "rating":
            all_tags.update(selected)

    result["tag"] = all_tags
    result["prediction"] = pred.astype(np.float32)
    return result


def get_oppai_oracle_tags(
    image_np: np.ndarray,
    session,
    csv_path: Path,
    preprocess_path: Path | None = None,
    threshold: float = 0.753,
    no_underline: bool = False,
    drop_overlap: bool = False,
) -> dict[str, dict[str, float] | np.ndarray]:
    preprocess = load_oppai_oracle_preprocess(preprocess_path)
    pixel_values, padding_mask = prepare_image_for_oppai_oracle(
        image_np,
        _target_size_from_session(session, preprocess),
        preprocess,
    )

    input_map = {}
    for input_info in session.get_inputs():
        if input_info.name == "pixel_values":
            input_map[input_info.name] = pixel_values
        elif input_info.name == "padding_mask":
            input_map[input_info.name] = padding_mask

    output_names = [output.name for output in session.get_outputs()]
    output_values = session.run(output_names, input_map)
    pred = output_values[0][0]
    if "probabilities" in output_names:
        pred = output_values[output_names.index("probabilities")][0]

    return postprocess_oppai_oracle(
        pred,
        csv_path=csv_path,
        threshold=threshold,
        no_underline=no_underline,
        drop_overlap=drop_overlap,
    )


def predict_oppai_oracle_tags(
    image_np: np.ndarray,
    session,
    csv_path: Path,
    preprocess_path: Path | None = None,
) -> dict[str, dict[str, float] | np.ndarray]:
    preprocess = load_oppai_oracle_preprocess(preprocess_path)
    pixel_values, padding_mask = prepare_image_for_oppai_oracle(
        image_np,
        _target_size_from_session(session, preprocess),
        preprocess,
    )

    input_map = {}
    for input_info in session.get_inputs():
        if input_info.name == "pixel_values":
            input_map[input_info.name] = pixel_values
        elif input_info.name == "padding_mask":
            input_map[input_info.name] = padding_mask

    output_names = [output.name for output in session.get_outputs()]
    output_values = session.run(output_names, input_map)
    pred = output_values[0][0]
    if "probabilities" in output_names:
        pred = output_values[output_names.index("probabilities")][0]

    tag_names, indexes_by_group = load_oppai_oracle_labels(csv_path, no_underline=False)
    result: dict[str, dict[str, float] | np.ndarray] = {}
    all_tags: dict[str, float] = {}

    for group_name, indexes in indexes_by_group.items():
        selected = {
            tag_names[index]: float(pred[index])
            for index in indexes
            if index < len(pred)
            and tag_names[index]
            and tag_names[index] not in {"<PAD>", "<UNK>"}
        }
        result[group_name] = selected
        if group_name != "rating":
            all_tags.update(selected)

    result["tag"] = all_tags
    result["prediction"] = pred.astype(np.float32)
    return result
