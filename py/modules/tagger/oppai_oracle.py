import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_LABEL_CACHE: dict[str, tuple[list[str], dict[str, list[int]]]] = {}
_PREPROCESS_CACHE: dict[str, dict] = {}


def load_oppai_oracle_labels(csv_path: Path) -> tuple[list[str], dict[str, list[int]]]:
    """Oppai Oracle の selected_tags.csv を読み込みます。

    CSV パスを受け取り、タグ名リストと rating/character/general のインデックス辞書を返します。

    Parameters
    ----------
    csv_path : Path
        タグ定義 CSV ファイルのパスです。

    Returns
    -------
    returns : tuple[list[str], dict[str, list[int]]]
        複数の処理結果をまとめたタプルを返します。
    """
    key = str(csv_path)
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
    """Oppai Oracle の前処理設定を読み込みます。

    preprocessing.json のパスまたは `None` を受け取り、画像サイズ、余白色、mean/std を含む辞書を返します。

    Parameters
    ----------
    preprocess_path : Path | None
        前処理設定 JSON ファイルのパスです。

    Returns
    -------
    returns : dict
        処理結果を格納した辞書を返します。
    """
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


def _target_size_from_session(session: Any, preprocess: dict) -> int:
    """ONNX セッションと前処理設定から入力画像サイズを決めます。

    ONNX セッションと前処理辞書を受け取り、`pixel_values` 入力 shape の空間サイズ、または設定上の `image_size` を返します。

    Parameters
    ----------
    session : Any
        ONNX Runtime の推論セッションです。
    preprocess : dict
        前処理設定を格納した辞書です。

    Returns
    -------
    returns : int
        計算または推定した整数値を返します。
    """
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
    """Oppai Oracle 用に画像と padding mask を前処理します。

    RGB `uint8 [H, W, 3]` 画像、入力サイズ、前処理設定を受け取り、`pixel_values [1, 3, H, W]` と `padding_mask [1, H, W]` を返します。

    Parameters
    ----------
    image_np : np.ndarray
        RGB 形式の uint8 画像配列です。
    target_size : int
        モデル入力に合わせる画像サイズです。
    preprocess : dict
        前処理設定を格納した辞書です。

    Returns
    -------
    returns : tuple[np.ndarray, np.ndarray]
        複数の処理結果をまとめたタプルを返します。
    """
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


def oppai_oracle_output_from_prediction(
    pred: np.ndarray,
    csv_path: Path,
) -> dict[str, dict[str, float] | np.ndarray]:
    """Oppai Oracle の予測ベクトルを全タグスコア辞書へ変換します。

    Parameters
    ----------
    pred : np.ndarray
        モデルが返したタグごとの確率ベクトルです。
    csv_path : Path
        タグ定義 CSV ファイルのパスです。

    Returns
    -------
    dict[str, dict[str, float] | np.ndarray]
        `tag` にしきい値未適用の全タグスコア、`prediction` に元の予測ベクトルを格納した辞書です。
    """
    tag_names, _indexes_by_group = load_oppai_oracle_labels(csv_path)

    tag_scores = {
        tag: float(score)
        for tag, score in zip(tag_names, pred.astype(float))
        if tag and tag not in {"<PAD>", "<UNK>"}
    }

    return {
        "tag": tag_scores,
        "prediction": pred.astype(np.float32),
    }


def predict_oppai_oracle_tags(
    image_np: np.ndarray,
    session: Any,
    csv_path: Path,
    preprocess_path: Path | None = None,
) -> dict[str, dict[str, float] | np.ndarray]:
    """Oppai Oracle を ONNX Runtime で推論し、全タグスコアを返します。

    RGB `uint8` 画像、ONNX セッション、CSV/preprocess パスを受け取り、しきい値未適用の全タグスコア辞書を返します。

    Parameters
    ----------
    image_np : np.ndarray
        RGB 形式の uint8 画像配列です。
    session : Any
        ONNX Runtime の推論セッションです。
    csv_path : Path
        タグ定義 CSV ファイルのパスです。
    preprocess_path : Path | None
        前処理設定 JSON ファイルのパスです。

    Returns
    -------
    returns : dict[str, dict[str, float] | np.ndarray]
        処理結果を格納した辞書を返します。
    """
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
    return oppai_oracle_output_from_prediction(pred, csv_path)


def predict_oppai_oracle_tags_tensorrt(
    image_np: np.ndarray,
    runner: Any,
    csv_path: Path,
    preprocess_path: Path | None = None,
) -> dict[str, dict[str, float] | np.ndarray]:
    """Oppai Oracle を TensorRT で推論し、全タグスコアを返します。

    RGB `uint8` 画像、TensorRT runner、CSV/preprocess パスを受け取り、しきい値未適用の全タグスコア辞書を返します。

    Parameters
    ----------
    image_np : np.ndarray
        RGB 形式の uint8 画像配列です。
    runner : Any
        TensorRT 推論 runner です。
    csv_path : Path
        タグ定義 CSV ファイルのパスです。
    preprocess_path : Path | None
        前処理設定 JSON ファイルのパスです。

    Returns
    -------
    returns : dict[str, dict[str, float] | np.ndarray]
        処理結果を格納した辞書を返します。
    """
    preprocess = load_oppai_oracle_preprocess(preprocess_path)
    target_size = int(preprocess.get("image_size", 448))
    for input_name in runner.input_names:
        if input_name == "pixel_values":
            shape = runner.engine.get_tensor_shape(input_name)
            int_dims = [int(dim) for dim in shape if int(dim) > 3]
            if int_dims:
                target_size = max(int_dims)

    pixel_values, padding_mask = prepare_image_for_oppai_oracle(image_np, target_size, preprocess)

    input_map = {}
    for input_name in runner.input_names:
        if input_name == "pixel_values":
            input_map[input_name] = pixel_values
        elif input_name == "padding_mask":
            input_map[input_name] = padding_mask

    output_values = runner.run(input_map, runner.output_names)
    pred = output_values[0][0]
    if "probabilities" in runner.output_names:
        pred = output_values[runner.output_names.index("probabilities")][0]
    return oppai_oracle_output_from_prediction(pred, csv_path)
