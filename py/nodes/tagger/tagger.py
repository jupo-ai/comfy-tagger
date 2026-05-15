from pathlib import Path

import comfy.utils
import folder_paths
import torch
from comfy_api.latest import io

from .common import PACKAGE_NAME, CATEGORY
from ...utils import mk_name
from ...modules import tagger as tagger_module
from ...modules.tagger import model_manager as tagger_model_manager


DEFAULT_MODELS_DIR = Path(__file__).parents[3] / "tagger_models"


def get_models_dir() -> Path:
    if "tagger" in folder_paths.folder_names_and_paths:
        paths = folder_paths.get_folder_paths("tagger")
        if paths:
            return Path(paths[0])
    return DEFAULT_MODELS_DIR


class Tagger(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "Tagger"),
            display_name="Tagger",
            category=CATEGORY,
            inputs=[
                io.Image.Input("images", tooltip="タグ付けする画像。複数画像の場合は各画像ごとにタグ文字列を出力する。"),
                io.Combo.Input("model", options=tagger_module.KNOWN_TAGGERS, tooltip="使用するTaggerモデル。未保存の場合は初回実行時に自動ダウンロードする。"),
                io.Combo.Input("inference_backend", options=["onnxruntime", "tensorrt"], tooltip="推論バックエンド。tensorrt は利用可能ならTensorRTを使い、失敗時はONNX Runtimeへフォールバックする。"),

                io.Boolean.Input("use_default_threshold", default=True, tooltip="有効な場合、各threshold入力ではなくモデル定義の推奨しきい値を使用する。未設定のカテゴリ推奨値はモデル定義のfallbackに従う。"),
                io.Boolean.Input("mcut_threshold", default=False, tooltip="有効な場合、各カテゴリのしきい値をMCut法で自動計算する。各threshold入力より優先される。"),

                io.Boolean.Input("include_general", default=True, tooltip="generalタグを出力に含める。"),
                io.Float.Input("threshold", default=0.35, min=0.0, max=1.0, step=0.01, tooltip="generalタグを採用する信頼度のしきい値。値を上げるほどタグ数が少なくなる。"),
                
                io.Boolean.Input("include_character", default=True, tooltip="characterタグを出力に含める。"),
                io.Float.Input("character_threshold", default=0.85, min=0.0, max=1.0, step=0.01, tooltip="characterタグを採用する信頼度のしきい値。キャラクター名などのタグに適用する。"),
                
                io.Boolean.Input("include_copyright", default=True, tooltip="copyrightタグを出力に含める。"),
                io.Float.Input("copyright_threshold", default=0.35, min=0.0, max=1.0, step=0.01, tooltip="copyrightタグを採用する信頼度のしきい値。"),

                io.Boolean.Input("include_artist", default=False, tooltip="artistタグを出力に含める。"),
                io.Float.Input("artist_threshold", default=0.35, min=0.0, max=1.0, step=0.01, tooltip="artistタグを採用する信頼度のしきい値。"),

                io.Boolean.Input("include_meta", default=False, tooltip="metaタグを出力に含める。"),
                io.Float.Input("meta_threshold", default=0.35, min=0.0, max=1.0, step=0.01, tooltip="metaタグを採用する信頼度のしきい値。"),
                
                io.Boolean.Input("include_rating", default=False, tooltip="ratingタグを出力に含める。"),
                io.Float.Input("rating_threshold", default=0.35, min=0.0, max=1.0, step=0.01, tooltip="ratingタグを採用する信頼度のしきい値。"),
                
                io.Boolean.Input("replace_underscore", default=True, tooltip="タグ内のアンダースコアをスペースに置換する。例: red_hair -> red hair"),
                io.Boolean.Input("drop_overlap", default=True, tooltip="重複・包含関係にあるタグを削除する。例: very_long_hair がある場合に long_hair を削除する。"),
                io.Boolean.Input("drop_blacklist", default=True, tooltip="imgutils互換のblacklistプリセットを使い、不要になりやすいタグを削除する。初回使用時にフィルタ用データを取得する。"),
                io.Boolean.Input("drop_basic_character", default=False, tooltip="red_hair や blue_eyes など、キャラクター外見の基本属性タグを削除する。"),
                io.Boolean.Input("escape_brackets", default=True, tooltip="タグ内の括弧をプロンプト用にエスケープする。"),
                io.Combo.Input("sort_mode", options=["original", "score", "shuffle"], tooltip="タグの並び順。original: モデル出力順、score: 信頼度順、shuffle: 人数タグ以外をランダム化。"),
                io.Boolean.Input("prioritize_people_tags", default=True, tooltip="solo や 1girl などの人数タグを先頭に寄せる。無効にすると sort_mode の結果をそのまま使う。"),
                io.Boolean.Input("trailing_comma", default=False, tooltip="出力文字列の末尾にもカンマとスペースを付ける。"),
                io.String.Input("exclude_tags", default="", tooltip="出力から除外するタグをカンマ区切りで指定する。例: watermark, signature"),
            ],
            outputs=[
                io.String.Output(is_output_list=True),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        model: str,
        inference_backend: str = "onnxruntime",
        
        use_default_threshold: bool = True,
        mcut_threshold: bool = False,
        
        include_general: bool = True,
        threshold: float = 0.35,
        
        include_character: bool = True,
        character_threshold: float = 0.85,
        
        include_copyright: bool = True,
        copyright_threshold: float = 0.35,

        include_artist: bool = False,
        artist_threshold: float = 0.35,

        include_meta: bool = False,
        meta_threshold: float = 0.35,
        
        include_rating: bool = False,
        rating_threshold: float = 0.35,
        
        replace_underscore: bool = True,
        drop_overlap: bool = True,
        drop_blacklist: bool = True,
        drop_basic_character: bool = False,
        escape_brackets: bool = True,
        sort_mode: str = "original",
        prioritize_people_tags: bool = True,
        trailing_comma: bool = False,
        exclude_tags: str = "",
    ):
        spec = tagger_module.get_model_spec(model)
        if use_default_threshold:
            threshold = spec.get_default_threshold()
            character_threshold = spec.get_default_character_threshold()
            artist_threshold = spec.get_default_artist_threshold()
            copyright_threshold = spec.get_default_copyright_threshold()
            meta_threshold = spec.get_default_meta_threshold()
            rating_threshold = spec.get_default_rating_threshold()

        models_dir = get_models_dir()
        download_pbar = comfy.utils.ProgressBar(1)
        model_paths = tagger_model_manager.prepare_model(model, models_dir, download_pbar)
        download_pbar.update_absolute(1, 1)

        inference_pbar = comfy.utils.ProgressBar(images.shape[0])
        results = []
        pp_options = tagger_module.TagPostprocessOptions(
            threshold=threshold,
            include_general=include_general,
            include_rating=include_rating,
            rating_threshold=rating_threshold,
            include_character=include_character,
            character_threshold=character_threshold,
            include_artist=include_artist,
            artist_threshold=artist_threshold,
            include_copyright=include_copyright,
            copyright_threshold=copyright_threshold,
            include_meta=include_meta,
            meta_threshold=meta_threshold,
            mcut_threshold=mcut_threshold,
            replace_underscore=replace_underscore,
            escape_brackets=escape_brackets,
            trailing_comma=trailing_comma,
            exclude_tags=exclude_tags,
            drop_overlap=drop_overlap,
            drop_blacklist=drop_blacklist,
            drop_basic_character=drop_basic_character,
            sort_mode=sort_mode,
            prioritize_people_tags=prioritize_people_tags,
        )
        for image in images:
            results.append(
                tagger_module.tag_image(
                    image=image,
                    model=model,
                    model_paths=model_paths,
                    inference_backend=inference_backend,
                    pp_options=pp_options,
                )
            )
            inference_pbar.update(1)

        return io.NodeOutput(results)
