import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_LABEL_CACHE: dict[str, tuple[list[str], list[int], list[int], list[int], list[int], list[int]]] = {}


def load_wd14_labels(csv_path: Path) -> tuple[list[str], list[int], list[int], list[int], list[int], list[int]]:
    """WD14 系 selected_tags.csv を読み込みます。

    CSV パスを受け取り、タグ名リストと rating/general/copyright/character/meta の各インデックスリストを返します。

    Parameters
    ----------
    csv_path : Path
        タグ定義 CSV ファイルのパスです。

    Returns
    -------
    returns : tuple[list[str], list[int], list[int], list[int], list[int], list[int]]
        複数の処理結果をまとめたタプルを返します。
    """
    key = str(csv_path)
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]

    tag_names = []
    rating_indexes = []
    general_indexes = []
    copyright_indexes = []
    character_indexes = []
    meta_indexes = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            tag = row["name"]
            tag_names.append(tag)

            category = int(row.get("category", -1))
            if category == 9:
                rating_indexes.append(index)
            elif category == 0:
                general_indexes.append(index)
            elif category == 3: # おそらく実際のcsvには存在しないが形式を揃えるため
                copyright_indexes.append(index)
            elif category == 4:
                character_indexes.append(index)
            elif category == 5: # おそらく実際のcsvには存在しないが形式を揃えるため
                meta_indexes.append(index)

    value = (tag_names, rating_indexes, general_indexes, copyright_indexes, character_indexes, meta_indexes)
    _LABEL_CACHE[key] = value
    return value


def prepare_image_for_wd14(image_np: np.ndarray, target_size: int) -> np.ndarray:
    """WD14 系モデル用に画像を前処理します。

    RGB `uint8 [H, W, 3]` 画像と入力サイズを受け取り、白背景で正方形化し BGR の `float32 [1, H, W, 3]` を返します。

    Parameters
    ----------
    image_np : np.ndarray
        RGB 形式の uint8 画像配列です。
    target_size : int
        モデル入力に合わせる画像サイズです。

    Returns
    -------
    returns : np.ndarray
        変換または推論で得た NumPy 配列を返します。
    """
    image = Image.fromarray(image_np, mode="RGB")
    max_dim = max(image.size)
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, ((max_dim - image.size[0]) // 2, (max_dim - image.size[1]) // 2))
    if max_dim != target_size:
        padded = padded.resize((target_size, target_size), Image.Resampling.BICUBIC)
    array = np.asarray(padded, dtype=np.float32)
    array = array[:, :, ::-1]
    return np.expand_dims(array, axis=0)


def wd14_output_from_prediction(
    pred: np.ndarray,
    csv_path: Path,
) -> dict[str, dict[str, float] | np.ndarray]:
    """WD14 の予測ベクトルを全タグスコア辞書へ変換します。

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
    tag_names, *_ = load_wd14_labels(csv_path)
    tag_score = {
        tag: float(score)
        for tag, score in zip(tag_names, pred.astype(float))
        if tag
    }
    return {
        "tag": tag_score, 
        "prediction": pred.astype(np.float32), 
    }
    


def predict_wd14_tags(
    image_np: np.ndarray,
    session: Any,
    csv_path: Path,
) -> dict[str, dict[str, float] | np.ndarray]:
    """WD14 を ONNX Runtime で推論し、全タグスコアを返します。

    RGB `uint8` 画像、ONNX セッション、CSV パスを受け取り、しきい値未適用の全タグスコア辞書を返します。

    Parameters
    ----------
    image_np : np.ndarray
        RGB 形式の uint8 画像配列です。
    session : Any
        ONNX Runtime の推論セッションです。
    csv_path : Path
        タグ定義 CSV ファイルのパスです。

    Returns
    -------
    returns : dict[str, dict[str, float] | np.ndarray]
        処理結果を格納した辞書を返します。
    """
    input_info = session.get_inputs()[0]
    target_size = int(input_info.shape[1])
    input_np = prepare_image_for_wd14(image_np, target_size)
    output_name = session.get_outputs()[0].name
    pred = session.run([output_name], {input_info.name: input_np})[0][0]
    return wd14_output_from_prediction(pred, csv_path)


def predict_wd14_tags_tensorrt(
    image_np: np.ndarray,
    runner: Any,
    csv_path: Path,
) -> dict[str, dict[str, float] | np.ndarray]:
    """WD14 を TensorRT で推論し、全タグスコアを返します。

    RGB `uint8` 画像、TensorRT runner、CSV パスを受け取り、しきい値未適用の全タグスコア辞書を返します。

    Parameters
    ----------
    image_np : np.ndarray
        RGB 形式の uint8 画像配列です。
    runner : Any
        TensorRT 推論 runner です。
    csv_path : Path
        タグ定義 CSV ファイルのパスです。

    Returns
    -------
    returns : dict[str, dict[str, float] | np.ndarray]
        処理結果を格納した辞書を返します。
    """
    input_name = runner.input_names[0]
    target_size = 448
    if len(runner.engine.get_tensor_shape(input_name)) >= 3:
        shape = runner.engine.get_tensor_shape(input_name)
        int_dims = [int(dim) for dim in shape if int(dim) > 3]
        if int_dims:
            target_size = max(int_dims)

    input_np = prepare_image_for_wd14(image_np, target_size)
    pred = runner.run({input_name: input_np})[0][0]
    return wd14_output_from_prediction(pred, csv_path)

