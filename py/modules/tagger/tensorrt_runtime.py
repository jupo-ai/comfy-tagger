from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

try:
    import tensorrt as trt

    HAS_TENSORRT = True
except Exception:  # pragma: no cover - depends on optional local runtime
    trt = None
    HAS_TENSORRT = False


TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if HAS_TENSORRT else None
_ENGINE_CACHE: dict[str, Any] = {}
_RUNNER_CACHE: dict[str, "TensorRTTaggerRunner"] = {}
_FAILED_ENGINES: set[str] = set()
_UNAVAILABLE_REASON_REPORTED: set[str] = set()


def _cli(message: str) -> None:
    """TensorRT 関連の CLI ログを出力します。

    表示するメッセージを受け取り、TaggerUI 用プレフィックス付きで標準出力へ出します。

    Parameters
    ----------
    message : str
        表示するログメッセージです。

    Returns
    -------
    returns : None
        値は返しません。
    """
    print(f"[TaggerUI][TensorRT] {message}", flush=True)


def tensorrt_available() -> bool:
    """TensorRT 推論を利用できるか判定します。

    引数は受け取らず、TensorRT パッケージと CUDA が利用可能なら `True` を返します。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    return HAS_TENSORRT and torch.cuda.is_available()


def tensorrt_unavailable_reason() -> str | None:
    """TensorRT が使えない理由を返します。

    引数は受け取らず、利用可能なら `None`、利用不能なら理由文字列を返します。

    Returns
    -------
    returns : str | None
        変換または生成した文字列を返します。
    """
    if not HAS_TENSORRT:
        return "tensorrt package is not importable"
    if not torch.cuda.is_available():
        return "CUDA is not available to PyTorch"
    return None


def engine_path_for_onnx(onnx_path: Path, use_fp16: bool = True) -> Path:
    """ONNX ファイルに対応する TensorRT engine パスを作ります。

    ONNX パスと FP16 使用フラグを受け取り、`.fp16.trt` または `.fp32.trt` の `Path` を返します。

    Parameters
    ----------
    onnx_path : Path
        変換元または実行元の ONNX ファイルパスです。
    use_fp16 : bool, optional
        TensorRT engine を FP16 で作成するかを指定します。

    Returns
    -------
    returns : Path
        対象ファイルまたはディレクトリのパスを返します。
    """
    suffix = ".fp16.trt" if use_fp16 else ".fp32.trt"
    return onnx_path.with_suffix(suffix)


def try_open_tensorrt_runner(
    onnx_path: Path,
    family: str,
    use_fp16: bool = True,
) -> "TensorRTTaggerRunner | None":
    """TensorRT runner を取得します。

    ONNX パス、モデルファミリ、FP16 使用フラグを受け取り、利用可能なら runner を返し、失敗時は `None` を返します。
    必要なら ONNX から engine を生成します。

    Parameters
    ----------
    onnx_path : Path
        変換元または実行元の ONNX ファイルパスです。
    family : str
        モデルファミリ名です。
    use_fp16 : bool, optional
        TensorRT engine を FP16 で作成するかを指定します。

    Returns
    -------
    returns : 'TensorRTTaggerRunner | None'
        TensorRT 推論 runner を返します。
    """
    unavailable_reason = tensorrt_unavailable_reason()
    if unavailable_reason is not None:
        if unavailable_reason not in _UNAVAILABLE_REASON_REPORTED:
            _cli(f"TensorRT is unavailable ({unavailable_reason}); using ONNX Runtime instead.")
            _UNAVAILABLE_REASON_REPORTED.add(unavailable_reason)
        return None

    engine_path = engine_path_for_onnx(onnx_path, use_fp16)
    key = str(engine_path)
    if key in _FAILED_ENGINES:
        _cli(f"Skipping TensorRT for {onnx_path.name}; a previous engine attempt failed. Using ONNX Runtime.")
        return None
    if key in _RUNNER_CACHE:
        _cli(f"Using cached TensorRT runner: {engine_path.name}")
        return _RUNNER_CACHE[key]

    try:
        if not engine_path.exists():
            _cli(f"No TensorRT engine found for {onnx_path.name}; starting conversion to {engine_path.name}.")
            build_tagger_engine(onnx_path, engine_path, family=family, use_fp16=use_fp16)
        else:
            _cli(f"Found TensorRT engine: {engine_path.name}")
        runner = TensorRTTaggerRunner(engine_path)
    except Exception as e:
        _cli(f"TensorRT is unavailable for {onnx_path.name} ({e}); using ONNX Runtime instead.")
        _FAILED_ENGINES.add(key)
        return None

    _cli(f"TensorRT runner is ready: {engine_path.name}")
    _RUNNER_CACHE[key] = runner
    return runner


def build_tagger_engine(
    onnx_path: Path,
    engine_path: Path,
    family: str,
    use_fp16: bool = True,
) -> None:
    """タグ付けモデルの ONNX を TensorRT engine へ変換します。

    ONNX パス、出力 engine パス、モデルファミリ、FP16 使用フラグを受け取り、変換済み engine ファイルを書き込みます。

    Parameters
    ----------
    onnx_path : Path
        変換元または実行元の ONNX ファイルパスです。
    engine_path : Path
        TensorRT engine ファイルのパスです。
    family : str
        モデルファミリ名です。
    use_fp16 : bool, optional
        TensorRT engine を FP16 で作成するかを指定します。

    Returns
    -------
    returns : None
        値は返しません。

    Raises
    ------
    ImportError
        必要な実行ライブラリを利用できない場合に送出されます。
    RuntimeError
        実行環境または推論処理が失敗した場合に送出されます。
    """
    if not HAS_TENSORRT:
        raise ImportError("TensorRT is not available")

    start = perf_counter()
    precision = "FP16" if use_fp16 else "FP32"
    _cli(f"Converting ONNX to TensorRT engine ({precision}): {onnx_path}")
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    _cli("Parsing ONNX graph...")
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("Failed to parse ONNX file: " + "; ".join(errors))
    _cli(f"ONNX parse completed. Inputs: {network.num_inputs}, outputs: {network.num_outputs}")

    profile = builder.create_optimization_profile()
    has_dynamic_input = False
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(int(dim) for dim in tensor.shape)
        if all(dim > 0 for dim in shape):
            continue
        profile_shape = _profile_shape_for_input(tensor.name, shape, family)
        profile.set_shape(tensor.name, profile_shape, profile_shape, profile_shape)
        has_dynamic_input = True
    if has_dynamic_input:
        config.add_optimization_profile(profile)

    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
    except Exception:
        pass

    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    _cli("Building TensorRT engine. This can take several minutes on the first run...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    elapsed = perf_counter() - start
    _cli(f"TensorRT engine saved: {engine_path} ({elapsed:.1f}s)")


def _profile_shape_for_input(name: str, shape: tuple[int, ...], family: str) -> tuple[int, ...]:
    """動的入力 shape 用の TensorRT 最適化 profile shape を返します。

    入力テンソル名、ONNX 上の shape、モデルファミリを受け取り、固定化した shape タプルを返します。

    Parameters
    ----------
    name : str
        対象テンソル名またはモデル名です。
    shape : tuple[int, ...]
        テンソル shape を表す整数タプルです。
    family : str
        モデルファミリ名です。

    Returns
    -------
    returns : tuple[int, ...]
        複数の処理結果をまとめたタプルを返します。
    """
    if all(dim > 0 for dim in shape):
        return shape

    if family == "wd14":
        return tuple(1 if dim < 0 and index == 0 else 448 if dim < 0 else dim for index, dim in enumerate(shape))

    if family == "oppai_oracle" and name == "padding_mask":
        return tuple(1 if index == 0 else 448 if dim < 0 else dim for index, dim in enumerate(shape))

    if len(shape) == 4:
        return tuple(1 if index == 0 else 3 if index == 1 and dim < 0 else 448 if dim < 0 else dim for index, dim in enumerate(shape))
    if len(shape) == 3:
        return tuple(1 if index == 0 else 448 if dim < 0 else dim for index, dim in enumerate(shape))
    return tuple(1 if dim < 0 else dim for dim in shape)


class TensorRTTaggerRunner:
    """TensorRT engine を読み込んでタグ付け推論を実行する runner です。

    engine ファイルパスを受け取り、入力名・出力名を保持し、NumPy 入力から NumPy 出力を返す `run` を提供します。

    Parameters
    ----------
    engine_path : Path
        TensorRT engine ファイルのパスです。

    Raises
    ------
    RuntimeError
        実行環境または推論処理が失敗した場合に送出されます。
    """

    def __init__(self, engine_path: Path) -> None:
        """TensorRT engine を読み込み、実行コンテキストと入出力名を初期化します。

        Parameters
        ----------
        engine_path : Path
            TensorRT engine ファイルのパスです。

        Returns
        -------
        returns : None
            値は返しません。

        Raises
        ------
        RuntimeError
            実行環境または推論処理が失敗した場合に送出されます。
        """
        self.engine_path = str(engine_path)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [
            name
            for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name
            for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]

    def _load_engine(self) -> Any:
        """engine ファイルをデシリアライズして TensorRT engine を返します。

        Returns
        -------
        returns : Any
            TensorRT engine オブジェクトを返します。

        Raises
        ------
        RuntimeError
            実行環境または推論処理が失敗した場合に送出されます。
        """
        if self.engine_path in _ENGINE_CACHE:
            return _ENGINE_CACHE[self.engine_path]

        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(Path(self.engine_path).read_bytes())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        _ENGINE_CACHE[self.engine_path] = engine
        return engine

    def run(self, input_map: dict[str, np.ndarray], output_names: list[str] | None = None) -> list[np.ndarray]:
        """TensorRT 推論を実行します。

        入力テンソル名から NumPy 配列への辞書と任意の出力名リストを受け取り、指定出力順の NumPy 配列リストを返します。

        Parameters
        ----------
        input_map : dict[str, np.ndarray]
            入力テンソル名から NumPy 配列への辞書です。
        output_names : list[str] | None
            取得する出力テンソル名のリストです。

        Returns
        -------
        returns : list[np.ndarray]
            処理結果のリストを返します。

        Raises
        ------
        RuntimeError
            実行環境または推論処理が失敗した場合に送出されます。
        """
        output_names = output_names or self.output_names
        device_tensors: dict[str, torch.Tensor] = {}

        for name in self.input_names:
            if name not in input_map:
                continue
            array = np.ascontiguousarray(input_map[name])
            tensor = torch.from_numpy(array).contiguous().to(device="cuda")
            device_tensors[name] = tensor
            self.context.set_input_shape(name, tuple(tensor.shape))

        for name in self.output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"TensorRT output shape is dynamic after input binding: {name} {shape}")
            dtype = _torch_dtype(self.engine.get_tensor_dtype(name))
            device_tensors[name] = torch.empty(shape, device="cuda", dtype=dtype)

        bindings = [0] * self.engine.num_io_tensors
        for index, name in enumerate(self.tensor_names):
            if name in device_tensors:
                bindings[index] = int(device_tensors[name].data_ptr())
                try:
                    self.context.set_tensor_address(name, bindings[index])
                except Exception:
                    pass

        if not self.context.execute_v2(bindings):
            raise RuntimeError("TensorRT inference failed")
        torch.cuda.synchronize()

        return [device_tensors[name].detach().cpu().numpy() for name in output_names]


def _torch_dtype(dtype: Any) -> torch.dtype:
    """TensorRT dtype を PyTorch dtype へ変換します。

    TensorRT の dtype を受け取り、対応する `torch.dtype` を返します。未知の型は `torch.float32` を返します。

    Parameters
    ----------
    dtype : Any
        変換対象の TensorRT dtype です。

    Returns
    -------
    returns : torch.dtype
        対応する PyTorch dtype を返します。
    """
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int8:
        return torch.int8
    if dtype == trt.bool:
        return torch.bool
    return torch.float32
