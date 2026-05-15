import csv
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_LABEL_CACHE: dict[str, tuple[list[str], dict[int, list[int]], dict[str, list[str]]]] = {}


def sigmoid(x: np.ndarray) -> np.ndarray:
    """NumPy 配列へ sigmoid を適用します。

    logits 配列を受け取り、0..1 範囲の確率配列を返します。

    Parameters
    ----------
    x : np.ndarray
        計算対象の NumPy 配列です。

    Returns
    -------
    returns : np.ndarray
        変換または推論で得た NumPy 配列を返します。
    """
    return 1.0 / (1.0 + np.exp(-x))


def load_pixai_labels(csv_path: Path) -> tuple[list[str], dict[int, list[int]], dict[str, list[str]]]:
    """PixAI selected_tags.csv を読み込みます。

    CSV パスを受け取り、タグ名リスト、カテゴリ番号別インデックス辞書、キャラクタータグから作品 IP への対応辞書を返します。

    Parameters
    ----------
    csv_path : Path
        タグ定義 CSV ファイルのパスです。

    Returns
    -------
    returns : tuple[list[str], dict[int, list[int]], dict[str, list[str]]]
        複数の処理結果をまとめたタプルを返します。
    """
    key = str(csv_path)
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]

    tag_names = []
    indexes_by_category: dict[int, list[int]] = {}
    ips_mapping: dict[str, list[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
        if records and "id" in records[0]:
            records.sort(key=lambda item: int(item.get("id") or 0))

        for index, row in enumerate(records):
            tag = row.get("name", "")
            tag_names.append(tag)

            if "category" in row and row.get("category") != "":
                category = int(row.get("category", 0) or 0)
            else:
                category = 4 if str(row.get("is_char", "")).lower() in {"1", "true", "yes"} else 0
            indexes_by_category.setdefault(category, []).append(index)

            ips = row.get("ips", "")
            if ips:
                try:
                    values = json.loads(ips)
                except json.JSONDecodeError:
                    try:
                        values = ast.literal_eval(ips)
                    except Exception:
                        values = []
                if values:
                    ips_mapping[tag] = values

    value = (tag_names, indexes_by_category, ips_mapping)
    _LABEL_CACHE[key] = value
    return value


def _target_size_from_session(session: Any) -> int:
    """PixAI ONNX セッションから入力画像サイズを推定します。

    ONNX セッションを受け取り、入力 shape から最大の空間サイズを返します。推定できない場合は 448 を返します。

    Parameters
    ----------
    session : Any
        ONNX Runtime の推論セッションです。

    Returns
    -------
    returns : int
        計算または推定した整数値を返します。
    """
    shape = session.get_inputs()[0].shape
    int_dims = [dim for dim in shape if isinstance(dim, int)]
    if len(int_dims) >= 2:
        return int(max(int_dims[-2:]))
    return 448


def _normalize_from_preprocess(preprocess_path: Path | None) -> tuple[list[float], list[float]]:
    """preprocess.json から正規化 mean/std を決めます。

    preprocess.json のパスまたは `None` を受け取り、RGB mean リストと std リストのタプルを返します。

    Parameters
    ----------
    preprocess_path : Path | None
        前処理設定 JSON ファイルのパスです。

    Returns
    -------
    returns : tuple[list[float], list[float]]
        複数の処理結果をまとめたタプルを返します。
    """
    if preprocess_path is None or not preprocess_path.exists():
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    try:
        data = json.loads(preprocess_path.read_text(encoding="utf-8"))
    except Exception:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    text = json.dumps(data).lower()
    if "imagenet" in text:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def prepare_image_for_pixai(image_np: np.ndarray, target_size: int, preprocess_path: Path | None = None) -> np.ndarray:
    """PixAI モデル用に画像を前処理します。

    RGB `uint8 [H, W, 3]` 画像、入力サイズ、任意の preprocess パスを受け取り、正規化済み `float32 [1, 3, H, W]` を返します。

    Parameters
    ----------
    image_np : np.ndarray
        RGB 形式の uint8 画像配列です。
    target_size : int
        モデル入力に合わせる画像サイズです。
    preprocess_path : Path | None
        前処理設定 JSON ファイルのパスです。

    Returns
    -------
    returns : np.ndarray
        変換または推論で得た NumPy 配列を返します。
    """
    image = Image.fromarray(image_np, mode="RGB")
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)

    array = np.asarray(image, dtype=np.float32) / 255.0
    mean, std = _normalize_from_preprocess(preprocess_path)
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    array = array.transpose(2, 0, 1)
    return np.expand_dims(array, axis=0).astype(np.float32)


def _prediction_from_outputs(output_names: list[str], output_values: list[np.ndarray]) -> np.ndarray:
    """PixAI の ONNX/TensorRT 出力から確率ベクトルを取り出します。

    出力名リストと出力配列リストを受け取り、`prediction` を優先し、logits の場合は sigmoid 済みの1次元配列を返します。

    Parameters
    ----------
    output_names : list[str]
        取得する出力テンソル名のリストです。
    output_values : list[np.ndarray]
        推論ランタイムから返された出力配列リストです。

    Returns
    -------
    returns : np.ndarray
        変換または推論で得た NumPy 配列を返します。
    """
    output_map = {name: value[0] for name, value in zip(output_names, output_values)}
    pred = output_map.get("prediction")
    if pred is None:
        pred = output_map.get("logits")
        if pred is None:
            pred = output_values[0][0]
        pred = sigmoid(pred)
    return pred


def pixai_output_from_prediction(
    pred: np.ndarray,
    csv_path: Path,
) -> dict[str, dict[str, float] | np.ndarray]:
    """PixAI の予測ベクトルを全タグスコア辞書へ変換します。

    確率ベクトルと CSV パスを受け取り、全タグスコアと `prediction` を含む辞書を返します。
    PixAI では character タグの IP 対応から copyright 相当タグのスコアも合成します。

    Parameters
    ----------
    pred : np.ndarray
        モデルが返したタグごとのスコアまたは確率ベクトルです。
    csv_path : Path
        タグ定義 CSV ファイルのパスです。
    Returns
    -------
    returns : dict[str, dict[str, float] | np.ndarray]
        処理結果を格納した辞書を返します。
    """
    tag_names, indexes_by_category, ips_mapping_all = load_pixai_labels(csv_path)
    
    tag_scores = {
        tag: float(score)
        for tag, score in zip(tag_names, pred.astype(float))
        if tag
    }
    
    # pixaiでは ip (copyright) のタグは character タグから決まるため
    # その copyright タグに所属する character タグの最大値を使う
    copyright_scores = {}
    for character_tag, ips in ips_mapping_all.items():
        character_score = tag_scores.get(character_tag)
        if character_score is None:
            continue
        for ip in ips:
            copyright_scores[ip] = max(
                copyright_scores.get(ip, float("-inf")), 
                character_score, 
            )
    tag_scores.update(copyright_scores)

    return {
        "tag": tag_scores, 
        "prediction": pred.astype(np.float32), 
    }

def predict_pixai_tags(
    image_np: np.ndarray,
    session: Any,
    csv_path: Path,
    thresholds_path: Path | None = None,
    preprocess_path: Path | None = None,
) -> dict[str, dict[str, float] | list[str] | np.ndarray]:
    """PixAI を ONNX Runtime で推論し、全タグスコアを返します。

    RGB `uint8` 画像、ONNX セッション、CSV/preprocess パスを受け取り、しきい値未適用の全タグスコア辞書を返します。

    Parameters
    ----------
    image_np : np.ndarray
        RGB 形式の uint8 画像配列です。
    session : Any
        ONNX Runtime の推論セッションです。
    csv_path : Path
        タグ定義 CSV ファイルのパスです。
    thresholds_path : Path | None
        互換性のために受け取ります。しきい値処理は共通後処理側で行うため、この関数内では使いません。
    preprocess_path : Path | None
        前処理設定 JSON ファイルのパスです。

    Returns
    -------
    returns : dict[str, dict[str, float] | list[str] | np.ndarray]
        処理結果を格納した辞書を返します。
    """
    input_info = session.get_inputs()[0]
    input_np = prepare_image_for_pixai(image_np, _target_size_from_session(session), preprocess_path)
    output_names = [output.name for output in session.get_outputs()]
    output_values = session.run(output_names, {input_info.name: input_np})
    pred = _prediction_from_outputs(output_names, output_values)
    return pixai_output_from_prediction(pred, csv_path)
    

def predict_pixai_tags_tensorrt(
    image_np: np.ndarray,
    runner: Any,
    csv_path: Path,
    thresholds_path: Path | None = None,
    preprocess_path: Path | None = None,
) -> dict[str, dict[str, float] | list[str] | np.ndarray]:
    """PixAI を TensorRT で推論し、全タグスコアを返します。

    RGB `uint8` 画像、TensorRT runner、CSV/preprocess パスを受け取り、しきい値未適用の全タグスコア辞書を返します。

    Parameters
    ----------
    image_np : np.ndarray
        RGB 形式の uint8 画像配列です。
    runner : Any
        TensorRT 推論 runner です。
    csv_path : Path
        タグ定義 CSV ファイルのパスです。
    thresholds_path : Path | None
        互換性のために受け取ります。しきい値処理は共通後処理側で行うため、この関数内では使いません。
    preprocess_path : Path | None
        前処理設定 JSON ファイルのパスです。

    Returns
    -------
    returns : dict[str, dict[str, float] | list[str] | np.ndarray]
        処理結果を格納した辞書を返します。
    """
    input_name = runner.input_names[0]
    shape = runner.engine.get_tensor_shape(input_name)
    int_dims = [int(dim) for dim in shape if int(dim) > 3]
    target_size = max(int_dims) if int_dims else 448
    input_np = prepare_image_for_pixai(image_np, target_size, preprocess_path)

    output_values = runner.run({input_name: input_np}, runner.output_names)
    pred = _prediction_from_outputs(runner.output_names, output_values)
    return pixai_output_from_prediction(pred, csv_path)
