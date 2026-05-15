from dataclasses import dataclass
from typing import Literal


TaggerFamily = Literal["wd14", "pixai", "oppai_oracle"]


@dataclass(frozen=True)
class ModelFile:
    """モデルリポジトリ内の1ファイル定義を表します。

    filename はリポジトリ上のパス、ext はローカル保存時の拡張子、required は必須ファイルかを受け取り、
    モデル準備時に必要なファイル一覧として参照されます。

    Parameters
    ----------
    filename : str
        リポジトリ上のファイルパスです。
    ext : str
        ローカル保存時に使う拡張子です。
    required : bool
        必須ファイルかどうかを示します。
    """

    filename: str
    ext: str
    required: bool = True


@dataclass(frozen=True)
class ModelSpec:
    """タグ付けモデル1件の仕様を表します。

    モデル名、ファミリ、Hugging Face リポジトリ、必要ファイル、カテゴリ別の既定しきい値を受け取り、
    ダウンロードや推論パイプラインが参照する不変のモデル定義として使われます。

    Parameters
    ----------
    name : str
        モデル名です。
    family : TaggerFamily
        モデルファミリです。
    repo_id : str
        Hugging Face 上のリポジトリ ID です。
    files : tuple[ModelFile, ...]
        モデルに必要なファイル定義のタプルです。
    default_threshold : float
        一般タグの既定しきい値です。
    default_character_threshold : float | None
        キャラクタータグの既定しきい値です。
    default_artist_threshold : float | None
        アーティストタグの既定しきい値です。
    default_rating_threshold : float | None
        レーティングタグの既定しきい値です。
    default_copyright_threshold : float | None
        版権タグの既定しきい値です。
    default_meta_threshold : float | None
        メタタグの既定しきい値です。
    """

    name: str
    family: TaggerFamily
    repo_id: str
    files: tuple[ModelFile, ...]
    default_threshold: float
    default_character_threshold: float | None = None
    default_artist_threshold: float | None = None
    default_rating_threshold: float | None = None
    default_copyright_threshold: float | None = None
    default_meta_threshold: float | None = None
    
    def get_default_threshold(self) -> float:
        """一般タグ用の既定しきい値を返します。

        Returns
        -------
        returns : float
            計算した浮動小数点値を返します。
        """
        return self.default_threshold

    def get_default_character_threshold(self) -> float:
        """キャラクタータグ用の既定しきい値を返します。

        個別値があればそれを返し、なければ一般タグ用の既定しきい値を返します。

        Returns
        -------
        returns : float
            計算した浮動小数点値を返します。
        """
        if self.default_character_threshold is not None:
            return self.default_character_threshold
        else:
            return self.get_default_threshold()

    def get_default_rating_threshold(self) -> float:
        """レーティングタグ用の既定しきい値を返します。

        個別値、キャラクターしきい値、一般しきい値の順にフォールバックします。

        Returns
        -------
        returns : float
            計算した浮動小数点値を返します。
        """
        if self.default_rating_threshold is not None:
            return self.default_rating_threshold
        else:
            return self.get_default_threshold()

    def get_default_artist_threshold(self) -> float:
        """アーティストタグ用の既定しきい値を返します。

        個別値があればそれを返し、なければ一般タグ用の既定しきい値を返します。

        Returns
        -------
        returns : float
            計算した浮動小数点値を返します。
        """
        if self.default_artist_threshold is not None:
            return self.default_artist_threshold
        else:
            return self.get_default_threshold()

    def get_default_copyright_threshold(self) -> float:
        """版権タグ用の既定しきい値を返します。

        個別値、キャラクターしきい値、一般しきい値の順にフォールバックします。

        Returns
        -------
        returns : float
            計算した浮動小数点値を返します。
        """
        if self.default_copyright_threshold is not None:
            return self.default_copyright_threshold
        else:
            return self.get_default_character_threshold()

    def get_default_meta_threshold(self) -> float:
        """メタタグ用の既定しきい値を返します。

        個別値があればそれを返し、なければ一般タグ用の既定しきい値を返します。

        Returns
        -------
        returns : float
            計算した浮動小数点値を返します。
        """
        if self.default_meta_threshold is not None:
            return self.default_meta_threshold
        else:
            return self.get_default_threshold()

_WD14_FILES = (
    ModelFile("model.onnx", "onnx"),
    ModelFile("selected_tags.csv", "csv"),
)

_PIXAI_FILES = (
    ModelFile("model.onnx", "onnx"),
    ModelFile("selected_tags.csv", "csv"),
    ModelFile("thresholds.csv", "thresholds.csv", required=False),
    ModelFile("preprocess.json", "preprocess.json", required=False),
)

_OPPAI_ORACLE_V1_1_FILES = (
    ModelFile("V1.1_onnx/model.onnx", "onnx"),
    ModelFile("V1.1_onnx/selected_tags.csv", "csv"),
    ModelFile("V1.1_onnx/preprocessing.json", "preprocessing.json"),
    ModelFile("V1.1_onnx/pr_thresholds.json", "pr_thresholds.json", required=False),
)


MODEL_SPECS: dict[str, ModelSpec] = {
    "wd-eva02-large-tagger-v3": ModelSpec(
        name="wd-eva02-large-tagger-v3",
        family="wd14",
        repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
        files=_WD14_FILES,
        default_threshold=0.35,
        default_character_threshold=0.85,
    ),
    "wd-vit-large-tagger-v3": ModelSpec(
        name="wd-vit-large-tagger-v3",
        family="wd14",
        repo_id="SmilingWolf/wd-vit-large-tagger-v3",
        files=_WD14_FILES,
        default_threshold=0.35,
        default_character_threshold=0.85,
    ),
    "wd-v1-4-swinv2-tagger-v2": ModelSpec(
        name="wd-v1-4-swinv2-tagger-v2",
        family="wd14",
        repo_id="SmilingWolf/wd-v1-4-swinv2-tagger-v2",
        files=_WD14_FILES,
        default_threshold=0.35,
        default_character_threshold=0.85,
    ),
    "wd-vit-tagger-v3": ModelSpec(
        name="wd-vit-tagger-v3",
        family="wd14",
        repo_id="SmilingWolf/wd-vit-tagger-v3",
        files=_WD14_FILES,
        default_threshold=0.35,
        default_character_threshold=0.85,
    ),
    "pixai-tagger-v0.9": ModelSpec(
        name="pixai-tagger-v0.9",
        family="pixai",
        repo_id="deepghs/pixai-tagger-v0.9-onnx",
        files=_PIXAI_FILES,
        default_threshold=0.30,
        default_character_threshold=0.85,
    ),
    "oppai-oracle-v1.1": ModelSpec(
        name="oppai-oracle-v1.1",
        family="oppai_oracle",
        repo_id="Grio43/OppaiOracle",
        files=_OPPAI_ORACLE_V1_1_FILES,
        default_threshold=0.753,
        default_character_threshold=0.753,
    ),
}

KNOWN_TAGGERS = list(MODEL_SPECS.keys())


def get_model_spec(model_name: str) -> ModelSpec:
    """モデル名から `ModelSpec` を取得します。

    登録済みモデル名を受け取り、対応するモデル仕様を返します。未知のモデル名なら `ValueError` を送出します。

    Parameters
    ----------
    model_name : str
        対象モデル名です。

    Returns
    -------
    returns : ModelSpec
        モデル仕様を返します。

    Raises
    ------
    ValueError
        入力値が想定外の場合に送出されます。
    """
    try:
        return MODEL_SPECS[model_name]
    except KeyError as e:
        raise ValueError(f"Unknown tagger model: {model_name}") from e
