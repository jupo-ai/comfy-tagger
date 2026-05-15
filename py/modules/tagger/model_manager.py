from pathlib import Path
from typing import Protocol

import requests
from tqdm import tqdm

from .registry import get_model_spec


class DownloadProgressLike(Protocol):
    """ダウンロード進捗通知に必要な最小インターフェースです。
    """

    def update_absolute(self, value: int, total: int) -> None:
        """現在のダウンロード済みバイト数と総バイト数を受け取り、進捗表示を更新します。

        Parameters
        ----------
        value : int
            `value` に渡す値です。
        total : int
            `total` に渡す値です。

        Returns
        -------
        returns : None
            値は返しません。
        """
        ...


def get_model_path(models_dir: Path, model_name: str, ext: str = "onnx") -> Path:
    """モデル名と拡張子からローカル保存先パスを作ります。

    モデルディレクトリ、モデル名、拡張子を受け取り、`<models_dir>/<model_name>.<ext>` の `Path` を返します。

    Parameters
    ----------
    models_dir : Path
        モデルファイルを保存するディレクトリです。
    model_name : str
        対象モデル名です。
    ext : str, optional
        `ext` に渡す値です。

    Returns
    -------
    returns : Path
        対象ファイルまたはディレクトリのパスを返します。
    """
    if ext.startswith("."):
        ext = ext.lstrip(".")
    return Path(models_dir) / f"{model_name}.{ext}"


def get_model_paths(model_name: str, models_dir: Path) -> dict[str, Path]:
    """モデルに関連するローカルファイルパス一覧を返します。

    モデル名とモデル保存ディレクトリを受け取り、リポジトリ上のファイル名および basename をキーにした `Path` 辞書を返します。

    Parameters
    ----------
    model_name : str
        対象モデル名です。
    models_dir : Path
        モデルファイルを保存するディレクトリです。

    Returns
    -------
    returns : dict[str, Path]
        処理結果を格納した辞書を返します。
    """
    spec = get_model_spec(model_name)
    paths: dict[str, Path] = {}
    for file in spec.files:
        path = get_model_path(models_dir, model_name, file.ext)
        if path.exists() or file.required:
            paths[file.filename] = path
            paths[Path(file.filename).name] = path
    return paths


def download_model(
    model_name: str,
    models_dir: Path,
    progress_bar: DownloadProgressLike | None = None,
) -> bool:
    """不足しているモデルファイルを Hugging Face からダウンロードします。

    モデル名、保存先ディレクトリ、任意の進捗通知先を受け取り、必須ファイルが揃えば `True`、失敗すれば `False` を返します。

    Parameters
    ----------
    model_name : str
        対象モデル名です。
    models_dir : Path
        モデルファイルを保存するディレクトリです。
    progress_bar : DownloadProgressLike | None
        ダウンロード進捗を通知する任意のオブジェクトです。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    spec = get_model_spec(model_name)
    base_url = f"https://huggingface.co/{spec.repo_id}/resolve/main"

    download_jobs = []
    for file in spec.files:
        dest = get_model_path(models_dir, model_name, file.ext)
        if not dest.exists():
            download_jobs.append((f"{base_url}/{file.filename}", dest, file.required))

    progress_state = None
    if progress_bar is not None and download_jobs:
        progress_state = {
            "downloaded": 0,
            "total": sum(_get_content_length(url) for url, _, _ in download_jobs),
        }

    for url, dest, required in download_jobs:
        if not _download_file(url, dest, progress_bar, progress_state) and required:
            return False

    return True


def prepare_model(
    model_name: str,
    models_dir: Path,
    progress_bar: DownloadProgressLike | None = None,
) -> dict[str, Path]:
    """モデルを利用可能な状態にしてファイルパス辞書を返します。

    モデル名と保存先を受け取り、必要に応じてダウンロードしたうえで `get_model_paths` の結果を返します。

    Parameters
    ----------
    model_name : str
        対象モデル名です。
    models_dir : Path
        モデルファイルを保存するディレクトリです。
    progress_bar : DownloadProgressLike | None
        ダウンロード進捗を通知する任意のオブジェクトです。

    Returns
    -------
    returns : dict[str, Path]
        処理結果を格納した辞書を返します。

    Raises
    ------
    RuntimeError
        実行環境または推論処理が失敗した場合に送出されます。
    """
    if not download_model(model_name, models_dir, progress_bar):
        raise RuntimeError(f"Failed to prepare tagger model: {model_name}")
    return get_model_paths(model_name, models_dir)


def _get_content_length(url: str) -> int:
    """URL の Content-Length を取得します。

    ダウンロード対象 URL を受け取り、取得できたバイト数を返します。取得不能な場合は 0 を返します。

    Parameters
    ----------
    url : str
        アクセス対象の URL です。

    Returns
    -------
    returns : int
        計算または推定した整数値を返します。
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        return int(response.headers.get("content-length", 0))
    except Exception:
        return 0


def _download_file(
    url: str,
    dest_path: Path,
    progress_bar: DownloadProgressLike | None = None,
    progress_state: dict[str, int] | None = None,
) -> bool:
    """単一ファイルをダウンロードして保存します。

    URL、保存先、任意の進捗通知情報を受け取り、保存に成功すれば `True`、失敗すれば不完全ファイルを削除して `False` を返します。

    Parameters
    ----------
    url : str
        アクセス対象の URL です。
    dest_path : Path
        ダウンロードしたファイルの保存先パスです。
    progress_bar : DownloadProgressLike | None
        ダウンロード進捗を通知する任意のオブジェクトです。
    progress_state : dict[str, int] | None
        複数ファイルの合算進捗を保持する辞書です。

    Returns
    -------
    returns : bool
        条件を満たす場合は True、満たさない場合は False を返します。
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with tqdm(
            total=total_size if total_size > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {dest_path.name}",
        ) as pbar:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue

                    f.write(chunk)
                    chunk_size = len(chunk)
                    downloaded += chunk_size
                    pbar.update(chunk_size)
                    if progress_bar is not None:
                        if progress_state is not None and progress_state["total"] > 0:
                            progress_state["downloaded"] += chunk_size
                            current = min(progress_state["total"], progress_state["downloaded"])
                            progress_bar.update_absolute(current, progress_state["total"])
                        elif total_size > 0:
                            current = min(total_size, downloaded)
                            progress_bar.update_absolute(current, total_size)

        return True

    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False
