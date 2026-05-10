# taggerモジュール詳細解説

`py/modules/tagger` は、画像タグ推定の中核モジュールです。
ONNX Runtimeによるモデル実行、モデルfamilyごとの前処理・後処理、タグカテゴリ判定、フィルタ、並べ替え、文字列化をまとめて扱います。

## 対象範囲

このモジュールは次の3つのfamilyを扱います。

- `wd14`: SmilingWolf系のWD tagger
- `pixai`: PixAI tagger
- `oppai_oracle`: OppaiOracle V1.1

family別の詳細は以下を参照してください。

- [WD14 family](tagger-wd14.md)
- [PixAI family](tagger-pixai.md)
- [OppaiOracle family](tagger-oppai-oracle.md)

## 公開API

`py/modules/tagger/__init__.py` は、外部コードから使う主要APIを再exportします。

主なAPIは次の通りです。

- `KNOWN_TAGGERS`: 利用可能なモデル名一覧
- `MODEL_SPECS`: モデル名から `ModelSpec` への定義表
- `get_model_spec(model_name)`: モデル定義の取得
- `prepare_model(model_name, models_dir, progress_bar=None)`: モデルファイルを準備し、推論に必要なパス辞書を返す
- `download_model(model_name, models_dir, progress_bar=None)`: モデルファイルをダウンロードする
- `get_model_paths(model_name, models_dir)`: ローカルモデルファイルのパス辞書を返す
- `open_onnx_session(model_path)`: ONNX Runtimeセッションの作成とキャッシュ
- `image_to_uint8(image)`: 1枚の画像テンソルをRGB uint8配列へ変換
- `images_to_uint8(images)`: 複数画像テンソルをRGB uint8配列へ変換する低レベル補助関数
- `predict_wd14_tags(...)`: WD14 familyの推論結果をタグスコア辞書へ変換
- `predict_pixai_tags(...)`: PixAI familyの推論結果をタグスコア辞書へ変換
- `predict_oppai_oracle_tags(...)`: OppaiOracle familyの推論結果をタグスコア辞書へ変換
- `get_wd14_tags(...)`: WD14 family単体の推論
- `get_pixai_tags(...)`: PixAI family単体の推論
- `get_oppai_oracle_tags(...)`: OppaiOracle family単体の推論
- `TagPostprocessOptions`: 統一後処理の設定
- `categorize_tags(...)`: 推論出力へしきい値、カテゴリ判定、フィルタ、並べ替えを適用
- `ordered_tags_from_groups(...)`: カテゴリ別タグを最終出力順へ統合
- `tags_to_prompt_text(...)`: タグ辞書をプロンプト文字列へ変換
- `tag_image(...)`: 1枚の画像をタグ文字列 `str` に変換
- `tag_image_grouped(...)`: 1枚の画像をカテゴリ別スコア辞書に変換

モデルダウンロードとファイルパス解決は `py/modules/tagger/model_manager.py` が担当します。
呼び出し側はモデル保存先ディレクトリを決め、`prepare_model(...)` で得た `model_paths` を `tag_image(...)` または `tag_image_grouped(...)` へ渡します。

## モデル定義

`registry.py` はモデル名、family、Hugging Face repository、必要ファイル、推奨しきい値を定義します。

`ModelSpec` は次の情報を持ちます。

- `name`: モデル名
- `family`: `wd14` / `pixai` / `oppai_oracle`
- `repo_id`: ダウンロード元のHugging Face repository
- `files`: 必要または任意のモデル関連ファイル
- `default_threshold`: general の推奨しきい値
- `default_character_threshold`: character の推奨しきい値。未設定なら `default_threshold` を使います。
- `default_rating_threshold`: rating の推奨しきい値。未設定なら `default_character_threshold`、それも未設定なら `default_threshold` を使います。
- `default_copyright_threshold`: copyright の推奨しきい値。未設定なら `default_character_threshold`、それも未設定なら `default_threshold` を使います。
- `default_meta_threshold`: meta の推奨しきい値。未設定なら `default_threshold` を使います。

カテゴリ別しきい値の取得には次のgetterを使います。

- `get_default_threshold()`: `default_threshold` を返します。
- `get_default_character_threshold()`
- `get_default_rating_threshold()`
- `get_default_copyright_threshold()`
- `get_default_meta_threshold()`

getterのfallback優先順位は次の通りです。

| getter | 優先順位 |
| --- | --- |
| `get_default_threshold()` | `default_threshold` |
| `get_default_character_threshold()` | `default_character_threshold` -> `default_threshold` |
| `get_default_rating_threshold()` | `default_rating_threshold` -> `default_character_threshold` -> `default_threshold` |
| `get_default_copyright_threshold()` | `default_copyright_threshold` -> `default_character_threshold` -> `default_threshold` |
| `get_default_meta_threshold()` | `default_meta_threshold` -> `default_threshold` |

`ModelFile` は次の情報を持ちます。

- `filename`: repository内のファイルパス
- `ext`: ローカル保存時の拡張子識別子
- `required`: ダウンロード失敗時に実行不能とみなすかどうか

ローカル保存名は `py/modules/tagger/model_manager.py` 側で `モデル名.ext` に変換されます。
たとえば `oppai-oracle-v1.1` の `V1.1_onnx/preprocessing.json` は、保存時には `oppai-oracle-v1.1.preprocessing.json` として扱われます。

## 対応モデル

現在の対応モデルは次の通りです。

| モデル名 | family | repo_id | general | character | copyright | meta | rating |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `wd-eva02-large-tagger-v3` | `wd14` | `SmilingWolf/wd-eva02-large-tagger-v3` | `0.35` | `0.85` | `0.85` | `0.35` | `0.85` |
| `wd-vit-large-tagger-v3` | `wd14` | `SmilingWolf/wd-vit-large-tagger-v3` | `0.35` | `0.85` | `0.85` | `0.35` | `0.85` |
| `wd-v1-4-swinv2-tagger-v2` | `wd14` | `SmilingWolf/wd-v1-4-swinv2-tagger-v2` | `0.35` | `0.85` | `0.85` | `0.35` | `0.85` |
| `wd-vit-tagger-v3` | `wd14` | `SmilingWolf/wd-vit-tagger-v3` | `0.35` | `0.85` | `0.85` | `0.35` | `0.85` |
| `pixai-tagger-v0.9` | `pixai` | `deepghs/pixai-tagger-v0.9-onnx` | `0.30` | `0.85` | `0.85` | `0.30` | `0.85` |
| `oppai-oracle-v1.1` | `oppai_oracle` | `Grio43/OppaiOracle` | `0.753` | `0.753` | `0.753` | `0.753` | `0.753` |

表の値はgetter適用後の実効値です。
呼び出し側がモデル推奨値を使いたい場合は、`get_default_threshold()` / `get_default_character_threshold()` / `get_default_copyright_threshold()` / `get_default_meta_threshold()` / `get_default_rating_threshold()` の戻り値を `tag_image(...)` へ渡します。
任意の値で実行したい場合は、各threshold引数へ直接値を渡します。

## 実行フロー

`tag_image(...)` の処理は大きく次の順番です。

1. `get_model_spec(model)` でモデル定義を取得します。
2. `open_onnx_session(model_paths["model.onnx"])` でONNX Runtimeセッションを取得します。
3. `image_to_uint8(image)` で1枚の画像テンソルを `np.uint8` のRGB配列へ変換します。
4. `_raw_output_for_image(...)` を呼び、familyごとの `predict_*_tags(...)` へ振り分けます。
5. `postprocess.categorize_tags(...)` でタグをカテゴリ化し、しきい値、include設定、フィルタ、並べ替えを適用します。
6. `postprocess.ordered_tags_from_groups(...)` でカテゴリ順に統合します。
7. `postprocess.tags_to_prompt_text(...)` でカンマ区切りの文字列へ変換します。

`tag_image_grouped(...)` は同じ推論・カテゴリ化処理を使いますが、最後に文字列化せず、カテゴリ別の `dict[str, dict[str, float]]` を返します。

taggerモジュール側の高レベルAPIは1枚処理を前提にしています。
複数画像を処理したい呼び出し側は、画像ごとに `tag_image(...)` または `tag_image_grouped(...)` を呼びます。

## モデルファイル準備

モデルファイルのダウンロード処理は `py/modules/tagger/model_manager.py` にあります。

主な関数は次の通りです。

- `get_model_path(models_dir, model_name, ext)`: 保存先ディレクトリと拡張子識別子からローカルファイルパスを作ります。
- `get_model_paths(model_name, models_dir)`: `ModelSpec.files` に基づいて推論関数へ渡すパス辞書を作ります。
- `download_model(model_name, models_dir, progress_bar=None)`: 不足しているモデル関連ファイルをHugging Faceから取得します。
- `prepare_model(model_name, models_dir, progress_bar=None)`: ダウンロードを実行し、成功時にパス辞書を返します。

このモジュールは保存先ディレクトリの決定を行いません。
呼び出し側が任意の `models_dir` を決め、その値を `prepare_model(...)` や `get_model_paths(...)` に渡します。

## 入力画像の扱い

`runtime.image_to_uint8(...)` は、1枚画像テンソル `[H, W, C]` を受け取ります。

処理内容は次の通りです。

- 3次元以外は `ValueError` にします。
- GPU上のテンソルでも `detach().cpu()` でCPUへ移します。
- 値域を `0.0` から `1.0` にclampします。
- 1チャンネル画像はRGB 3チャンネルへ展開します。
- 4チャンネル以上の画像は先頭3チャンネルだけを使います。
- 2チャンネルなど不正なチャンネル数は `ValueError` にします。
- 最終的に `0..255` の `np.uint8` へ変換します。

この段階ではモデル固有のresize、padding、正規化は行いません。
それらは `wd14.py` / `pixai.py` / `oppai_oracle.py` に分かれています。

`images_to_uint8(...)` は複数画像用の低レベル補助関数として残っていますが、通常のタグ付けpipelineは `image_to_uint8(...)` を使います。

## ONNX Runtimeセッション

`runtime.open_onnx_session(...)` は、モデルパス文字列をキーに `InferenceSession` をキャッシュします。

providerは次の優先順で選ばれます。

1. `CUDAExecutionProvider`
2. `CPUExecutionProvider`
3. ONNX Runtimeが返すその他の利用可能provider

同じモデルを複数画像や複数回の実行で使う場合、セッション作成コストを避けるため、既存セッションが再利用されます。

## family振り分け

`pipeline._raw_output_for_image(...)` は `ModelSpec.family` を見て推論関数を選びます。

- `wd14` -> `predict_wd14_tags(...)`
- `pixai` -> `predict_pixai_tags(...)`
- `oppai_oracle` -> `predict_oppai_oracle_tags(...)`

`predict_*_tags(...)` はモデルfamily固有の画像前処理、ONNX推論、ラベルCSVとの対応付けに集中します。
しきい値、アンダースコア置換、blacklist、overlap、並べ替え、文字列化は `postprocess.py` 側で統一的に扱います。

## 出力構造

familyごとの推論関数は、おおむね次のキーを含む辞書を返します。

- `rating`: ratingタグとスコア
- `general`: generalタグとスコア
- `character`: characterタグとスコア
- `copyright`: copyrightタグとスコア
- `meta`: metaタグとスコア
- `tag`: rating以外をまとめたタグ
- `prediction`: モデルの生に近い予測配列

ただし、すべてのfamilyが常に全カテゴリを返すとは限りません。
`postprocess.categorize_tags(...)` は `rating` / `character` / `copyright` / `meta` / `general` / `tag` を順に見て、存在する辞書だけを統合します。

カテゴリが明示されているタグはそのカテゴリを優先します。
明示カテゴリがない場合は、次の判定で補完します。

- `rating:` prefixがあるタグは `rating`
- `jupo-ai/danbooru-characters` 由来のcharacter一覧に含まれるタグは `character`
- 同dataset由来のcopyright一覧に含まれるタグは `copyright`
- 同dataset由来のmeta一覧に含まれるタグは `meta`
- それ以外は `general`

## しきい値とinclude設定

`postprocess.categorize_tags(...)` はカテゴリごとに別のしきい値とinclude設定を持ちます。

| カテゴリ | しきい値 | include設定 |
| --- | --- | --- |
| `rating` | `rating_threshold` | `include_rating` |
| `character` | `character_threshold` | `include_character` |
| `copyright` | `copyright_threshold` | `include_copyright` |
| `meta` | `meta_threshold` | `include_meta` |
| `general` | `threshold` | 常に有効 |

スコア判定は `score > threshold` です。
しきい値と同じ値のタグは採用されません。

## タグ除外

タグ除外はカテゴリごとに適用されます。

処理順は次の通りです。

1. `exclude_tags`
2. `drop_blacklist`
3. `drop_basic_character`
4. `sort_mode`

`exclude_tags` はカンマ区切り文字列です。
大文字小文字を区別せず、完全一致したタグを除外します。

`drop_blacklist` は `alea31415/tag_filtering` の `blacklist_tags.txt` を初回使用時に取得し、不要になりやすいタグを除外します。

`drop_basic_character` は、髪色・目色などの基本的なキャラクター外見属性を除外するためのルールベースフィルタです。

## 重複タグ除去

`drop_overlap` が有効な場合、`overlap.py` の `drop_overlap_tags(...)` がgeneralタグに適用されます。

この処理は2種類の重複を落とします。

- `alea31415/tag_filtering` の `overlap_tags_simplified.json` に基づく包含関係
- `long hair` と `very long hair` のように、片方のタグ文字列が別タグに含まれる関係

`postprocess.categorize_tags(...)` では、カテゴリ化後の `general` に対して適用します。
family単体の後処理にも `drop_overlap` 引数はありますが、pipeline APIでは `postprocess.py` 側で一元的に処理します。

## 並べ替え

`order.sort_tags(...)` は次の `sort_mode` を扱います。

- `original`: モデルまたはCSVの順序を保ちます。
- `score`: スコア降順にします。
- `shuffle`: 人数タグ以外をランダムに並べ替えます。

`prioritize_people_tags` が有効な場合、`solo` と `1girl` / `2girls` / `1boy` / `2boys` / `3+girls` のような人数タグを先頭へ寄せます。

この並べ替えはカテゴリごとに行われます。
最終的なカテゴリ結合順は固定で、`character` -> `copyright` -> `general` -> `meta` -> `rating` です。

## 文字列化

`postprocess.tags_to_prompt_text(...)` は、タグ辞書をカンマ区切りの文字列へ変換します。

`tag_image(...)` では次の設定で呼ばれます。

- `use_escape=True`
- `include_score=False`
- `score_descend=False`
- `trailing_comma` は `tag_image(...)` の引数に従う

そのため、タグ名はpipelineで整えた形を維持し、括弧やバックスラッシュはプロンプト用にescapeされます。
`trailing_comma=True` の場合、空でない出力末尾に `, ` が追加されます。

## アンダースコア置換

`tag_image(...)` / `tag_image_grouped(...)` の `replace_underscore` は、`postprocess.categorize_tags(...)` でタグ名を整形するときに使われます。

有効な場合、推論出力に含まれるタグ名の `_` をスペースへ置換します。
ただし、`0_0` や `>_<` など一部の顔文字タグは壊さないように例外扱いされます。

## 外部データとキャッシュ

補助フィルタやカテゴリ判定ではHugging Face上のdatasetを初回取得します。

| 用途 | repository | 保存先 |
| --- | --- | --- |
| character / copyright / meta 判定 | `jupo-ai/danbooru-characters` | `tagger_models/danbooru_characters` |
| blacklist / overlap | `alea31415/tag_filtering` | `tagger_models/tag_filtering` |

各モジュールは `lru_cache` やモジュール内辞書で読み込み結果をキャッシュします。
一度取得済みのファイルが存在する場合は、再ダウンロードせずローカルファイルを使います。

## 外部コード利用

taggerモジュールの実装には、[deepghs/imgutils](https://github.com/deepghs/imgutils) のコードを参考・利用している部分があります。

## エラーになりやすい条件

次の条件では例外または実行失敗が起こります。

- 未知のモデル名を渡した場合、`get_model_spec(...)` が `ValueError` を投げます。
- 入力画像テンソルが `[H, W, C]` でない場合、`image_to_uint8(...)` が `ValueError` を投げます。
- チャンネル数が1、3、4以上のいずれでもない場合、`image_to_uint8(...)` が `ValueError` を投げます。
- 必須モデルファイルのダウンロードに失敗した場合、`prepare_model(...)` が `RuntimeError` を投げます。
- 初回の外部dataset取得に失敗した場合、blacklist、overlap、character/copyright/meta判定の処理で例外が起こる可能性があります。

## 新しいモデルfamilyを追加する場合

新しいfamilyを追加する場合は、最低限次の変更が必要です。

1. `registry.TaggerFamily` にfamily名を追加します。
2. `registry.MODEL_SPECS` に `ModelSpec` を追加します。
3. family専用の `get_xxx_tags(...)` を実装します。
4. `pipeline._raw_output_for_image(...)` に分岐を追加します。
5. `__init__.py` から必要な関数をexportします。
6. 公開APIや利用側のモデル選択リストがある場合は、必要に応じてモデル名を追加します。

推論関数は、可能な限り `rating` / `general` / `copyright` / `meta` / `character` / `tag` / `prediction` の構造にそろえると、pipeline側のカテゴリ処理をそのまま利用できます。
