from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


_SESSIONS: dict[str, ort.InferenceSession] = {}


def open_onnx_session(model_path: Path) -> ort.InferenceSession:
    """ONNX Runtime の推論セッションを開きます。

    ONNX ファイルパスを受け取り、利用可能な CUDA/CPU provider を設定した `InferenceSession` を返します。
    同じパスはキャッシュ済みセッションを再利用します。

    Parameters
    ----------
    model_path : Path
        `model_path` に渡す値です。

    Returns
    -------
    returns : ort.InferenceSession
        ONNX Runtime の推論セッションを返します。
    """
    key = str(model_path)
    if key in _SESSIONS:
        return _SESSIONS[key]

    providers = [
        provider
        for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider in ort.get_available_providers()
    ]
    if not providers:
        providers = ort.get_available_providers()

    session = ort.InferenceSession(key, providers=providers)
    _SESSIONS[key] = session
    return session


def images_to_uint8(images: torch.Tensor) -> np.ndarray:
    """複数画像テンソルを `uint8` の NumPy 配列へ変換します。

    `[B, H, W, C]` 形状で 0..1 範囲の `torch.Tensor` を受け取り、`uint8 [B, H, W, 3]` を返します。

    Parameters
    ----------
    images : torch.Tensor
        0..1 範囲の画像バッチテンソルです。

    Returns
    -------
    returns : np.ndarray
        変換または推論で得た NumPy 配列を返します。

    Raises
    ------
    ValueError
        入力値が想定外の場合に送出されます。
    """
    if images.dim() != 4:
        raise ValueError(f"Expected images [B, H, W, C], got {images.dim()}D (shape: {images.shape})")

    images = images.detach().cpu().clamp(0.0, 1.0)
    if images.shape[-1] == 1:
        images = images.expand(*images.shape[:-1], 3)
    elif images.shape[-1] > 3:
        images = images[..., :3]
    elif images.shape[-1] != 3:
        raise ValueError(f"Expected image channels 1, 3, or 4, got shape: {images.shape}")

    return (images.numpy() * 255.0).round().astype(np.uint8)


def image_to_uint8(image: torch.Tensor) -> np.ndarray:
    """単一画像テンソルを `uint8` の NumPy 配列へ変換します。

    `[H, W, C]` 形状で 0..1 範囲の `torch.Tensor` を受け取り、`uint8 [H, W, 3]` を返します。

    Parameters
    ----------
    image : torch.Tensor
        0..1 範囲の画像テンソルです。

    Returns
    -------
    returns : np.ndarray
        変換または推論で得た NumPy 配列を返します。

    Raises
    ------
    ValueError
        入力値が想定外の場合に送出されます。
    """
    if image.dim() != 3:
        raise ValueError(f"Expected image [H, W, C], got {image.dim()}D (shape: {image.shape})")
    return images_to_uint8(image.unsqueeze(0))[0]
