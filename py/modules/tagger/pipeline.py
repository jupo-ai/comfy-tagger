from pathlib import Path
from typing import Any

import numpy as np
import torch

from .format import tags_to_text
from .postprocess import TagPostprocessOptions, execute_postprocess, ordered_tags_from_groups
from .registry import ModelSpec, get_model_spec
from .runtime import image_to_uint8, open_onnx_session
from .tensorrt_runtime import try_open_tensorrt_runner
from .wd14 import predict_wd14_tags, predict_wd14_tags_tensorrt
from .pixai import predict_pixai_tags, predict_pixai_tags_tensorrt
from .oppai_oracle import predict_oppai_oracle_tags, predict_oppai_oracle_tags_tensorrt


_ONNXRUNTIME_SELECTION_REPORTED: set[str] = set()


def _raw_output_for_image(
    image_np: np.ndarray,
    spec: ModelSpec,
    session: Any,
    model_paths: dict[str, Path],
) -> dict:
    """ONNX Runtime でモデル別の生タグスコア出力を取得します。

    `uint8` 画像、モデル仕様、ONNX セッション、モデル関連パスを受け取り、モデル別 `predict_*` の出力辞書を返します。

    Parameters
    ----------
    image_np : np.ndarray
        RGB 形式の uint8 画像配列です。
    spec : ModelSpec
        利用するモデル仕様です。
    session : Any
        ONNX Runtime の推論セッションです。
    model_paths : dict[str, Path]
        モデル関連ファイル名からローカルパスへの辞書です。

    Returns
    -------
    returns : dict
        処理結果を格納した辞書を返します。

    Raises
    ------
    ValueError
        入力値が想定外の場合に送出されます。
    """
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


def _raw_output_for_image_tensorrt(
    image_np: np.ndarray,
    spec: ModelSpec,
    runner: Any,
    model_paths: dict[str, Path],
) -> dict:
    """TensorRT でモデル別の生タグスコア出力を取得します。

    `uint8` 画像、モデル仕様、TensorRT runner、モデル関連パスを受け取り、モデル別 TensorRT `predict_*` の出力辞書を返します。

    Parameters
    ----------
    image_np : np.ndarray
        RGB 形式の uint8 画像配列です。
    spec : ModelSpec
        利用するモデル仕様です。
    runner : Any
        TensorRT 推論 runner です。
    model_paths : dict[str, Path]
        モデル関連ファイル名からローカルパスへの辞書です。

    Returns
    -------
    returns : dict
        処理結果を格納した辞書を返します。

    Raises
    ------
    ValueError
        入力値が想定外の場合に送出されます。
    """
    if spec.family == "wd14":
        return predict_wd14_tags_tensorrt(
            image_np,
            runner,
            csv_path=model_paths["selected_tags.csv"],
        )
    if spec.family == "pixai":
        return predict_pixai_tags_tensorrt(
            image_np,
            runner,
            csv_path=model_paths["selected_tags.csv"],
            thresholds_path=model_paths.get("thresholds.csv"),
            preprocess_path=model_paths.get("preprocess.json"),
        )
    if spec.family == "oppai_oracle":
        return predict_oppai_oracle_tags_tensorrt(
            image_np,
            runner,
            csv_path=model_paths["selected_tags.csv"],
            preprocess_path=model_paths.get("preprocessing.json"),
        )
    raise ValueError(f"Unsupported tagger family: {spec.family}")


def tag_image_grouped(
    image: torch.Tensor,
    model: str,
    model_paths: dict[str, Path],
    inference_backend: str = "onnxruntime",
    pp_options: TagPostprocessOptions | None = None,
) -> dict[str, dict[str, float]]:
    """画像をタグ付けし、カテゴリ別タグスコア辞書を返します。

    画像テンソル、モデル名、モデルファイルパス、推論バックエンド、後処理設定を受け取り、
    共通後処理を適用した `artist/rating/character/copyright/meta/general` ごとのタグスコア辞書を返します。

    Parameters
    ----------
    image : torch.Tensor
        0..1 範囲の画像テンソルです。
    model : str
        利用するタグ付けモデル名です。
    model_paths : dict[str, Path]
        モデル関連ファイル名からローカルパスへの辞書です。
    inference_backend : str, optional
        `"onnxruntime"` なら ONNX Runtime を使い、それ以外なら TensorRT を試して失敗時に ONNX Runtime へ戻します。
    pp_options : TagPostprocessOptions | None, optional
        しきい値、カテゴリ出力、除外、整形、並び替えをまとめた後処理設定です。`None` なら既定値を使います。

    Returns
    -------
    returns : dict[str, dict[str, float]]
        処理結果を格納した辞書を返します。
    """
    spec = get_model_spec(model)
    image_np = image_to_uint8(image)
    if inference_backend == "onnxruntime":
        if model not in _ONNXRUNTIME_SELECTION_REPORTED:
            print(f"[TaggerUI][ONNXRuntime] Using ONNX Runtime for {model}; TensorRT is disabled by settings.", flush=True)
            _ONNXRUNTIME_SELECTION_REPORTED.add(model)
        session = open_onnx_session(model_paths["model.onnx"])
        output = _raw_output_for_image(image_np, spec, session, model_paths)
    else:
        runner = try_open_tensorrt_runner(model_paths["model.onnx"], family=spec.family)
        if runner is not None:
            try:
                output = _raw_output_for_image_tensorrt(image_np, spec, runner, model_paths)
            except Exception as e:
                print(f"[TaggerUI][TensorRT] Inference failed ({e}); using ONNX Runtime instead.", flush=True)
                session = open_onnx_session(model_paths["model.onnx"])
                output = _raw_output_for_image(image_np, spec, session, model_paths)
        else:
            session = open_onnx_session(model_paths["model.onnx"])
            output = _raw_output_for_image(image_np, spec, session, model_paths)
    
    if pp_options is None:
        pp_options = TagPostprocessOptions()
    
    return execute_postprocess(output, pp_options)


def tag_image(
    image: torch.Tensor,
    model: str,
    model_paths: dict[str, Path],
    inference_backend: str = "onnxruntime",
    pp_options: TagPostprocessOptions | None = None,
) -> str:
    """画像をタグ付けし、プロンプト文字列を返します。

    `tag_image_grouped()` の結果を `artist/character/copyright/general/meta/rating` の順に平坦化し、
    同じ順序でカンマ区切りプロンプトへ変換して返します。

    Parameters
    ----------
    image : torch.Tensor
        0..1 範囲の `[H, W, C]` 画像テンソルです。
    model : str
        利用するタグ付けモデル名です。
    model_paths : dict[str, Path]
        モデル関連ファイル名からローカルパスへの辞書です。
    inference_backend : str, optional
        使用する推論バックエンドです。
    pp_options : TagPostprocessOptions | None, optional
        後処理設定です。`None` なら既定値を使います。

    Returns
    -------
    str
        カンマ区切りのプロンプト文字列です。
    """
    if pp_options is None:
        pp_options = TagPostprocessOptions()
    grouped = tag_image_grouped(image, model, model_paths, inference_backend, pp_options)
    tags = ordered_tags_from_groups(grouped)
    return tags_to_text(
        tags,
        use_spaces=False,
        use_escape=False,
        include_score=False,
        score_descend=False,
        trailing_comma=pp_options.trailing_comma,
    )

