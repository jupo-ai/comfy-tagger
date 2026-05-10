from dataclasses import dataclass
from typing import Literal


TaggerFamily = Literal["wd14", "pixai", "oppai_oracle"]


@dataclass(frozen=True)
class ModelFile:
    filename: str
    ext: str
    required: bool = True


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: TaggerFamily
    repo_id: str
    files: tuple[ModelFile, ...]
    default_threshold: float
    default_character_threshold: float | None = None
    default_rating_threshold: float | None = None
    default_copyright_threshold: float | None = None
    default_meta_threshold: float | None = None
    
    def get_default_threshold(self) -> float:
        return self.default_threshold

    def get_default_character_threshold(self) -> float:
        """
        優先度: 
        1 -> default_character_threshold
        2 -> default_threshold
        """
        if self.default_character_threshold is not None:
            return self.default_character_threshold
        else:
            return self.get_default_threshold()

    def get_default_rating_threshold(self) -> float:
        """
        優先度: 
        1 -> default_rating_threshold
        2 -> default_character_threshold
        3 -> default_threshold
        """
        if self.default_rating_threshold is not None:
            return self.default_rating_threshold
        else:
            return self.get_default_character_threshold()

    def get_default_copyright_threshold(self) -> float:
        """
        優先度:
        1 -> default_copyright_threshold
        2 -> default_character_threshold
        3 -> default_threshold
        """
        if self.default_copyright_threshold is not None:
            return self.default_copyright_threshold
        else:
            return self.get_default_character_threshold()

    def get_default_meta_threshold(self) -> float:
        """
        優先度:
        1 -> default_meta_threshold
        2 -> default_threshold
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
    try:
        return MODEL_SPECS[model_name]
    except KeyError as e:
        raise ValueError(f"Unknown tagger model: {model_name}") from e
