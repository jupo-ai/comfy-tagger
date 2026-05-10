# OppaiOracle family詳細解説

OppaiOracle familyは、`Grio43/OppaiOracle` のV1.1 ONNX版を扱う実装です。
実装本体は `py/modules/tagger/oppai_oracle.py` にあります。

## 対応モデル

現在のOppaiOracle familyには次のモデルが登録されています。

| モデル名 | repo_id |
| --- | --- |
| `oppai-oracle-v1.1` | `Grio43/OppaiOracle` |

必要ファイルと任意ファイルは次の通りです。

| repository内ファイル | ローカル識別子 | 必須 |
| --- | --- | --- |
| `V1.1_onnx/model.onnx` | `onnx` | yes |
| `V1.1_onnx/selected_tags.csv` | `csv` | yes |
| `V1.1_onnx/preprocessing.json` | `preprocessing.json` | yes |
| `V1.1_onnx/pr_thresholds.json` | `pr_thresholds.json` | no |

`ModelSpec` のgetterで取得できる実効しきい値は次の通りです。

- general: `0.753`
- character: `0.753`
- copyright: `0.753`
- meta: `0.753`
- rating: `0.753`

`ModelSpec` では `default_threshold=0.753`、`default_character_threshold=0.753` を設定し、copyright/rating/metaの個別値は省略しています。
そのため、copyright/ratingは `default_character_threshold` へfallbackします。
metaは `default_threshold` へfallbackします。

現在の実装では `pr_thresholds.json` はダウンロード対象に含まれますが、後処理では直接使われていません。

## ラベル読み込み

`load_oppai_oracle_labels(csv_path, no_underline=False)` は `selected_tags.csv` を読み込み、次の2要素を返します。

1. `tag_names`
2. `indexes_by_group`

CSVに `id` 列がある場合、`id` 昇順に並べ替えてからindexを割り当てます。

タグ名は次の列を順に見て取得します。

1. `name`
2. `tag`
3. `tag_name`
4. いずれもなければ空文字

カテゴリは `category` 列から取得します。
空の場合は `0` として扱います。

カテゴリ値は次のように解釈されます。

| category | グループ |
| --- | --- |
| `9` | `rating` |
| `4` | `character` |
| その他 | `general` |

WD14と異なり、`category=3` のcopyright専用分岐はありません。
CSV内にcopyright/meta相当のタグがあっても、OppaiOracle単体後処理では基本的に `general` として扱われます。
pipeline側では `jupo-ai/danbooru-characters` のcopyright/meta一覧によって再分類される可能性があります。

## preprocessing.json

`load_oppai_oracle_preprocess(preprocess_path)` は前処理設定をJSONから読み込みます。

存在しない場合や読み込みに失敗した場合は次の既定値を使います。

```json
{
  "image_size": 448,
  "pad_color_rgb": [114, 114, 114],
  "normalize_mean": [0.5, 0.5, 0.5],
  "normalize_std": [0.5, 0.5, 0.5]
}
```

読み込んだJSONに一部キーが欠けている場合も、同じ既定値で補完されます。
結果は `_PREPROCESS_CACHE` に保存され、同じファイルは再読み込みされません。

## 前処理

`prepare_image_for_oppai_oracle(image_np, target_size, preprocess)` は、画像テンソルとpadding maskを作ります。

処理内容は次の通りです。

1. `np.ndarray` からPillowのRGB画像を作ります。
2. 元画像の縦横比を維持したまま、`target_size` 内に収まるscaleを計算します。
3. BICUBICでresizeします。
4. `target_size x target_size` のRGBキャンバスを作ります。
5. 背景色は `preprocess["pad_color_rgb"]` です。既定値は `[114, 114, 114]` です。
6. resize後の画像を中央に貼り付けます。
7. padding部分が `True`、画像部分が `False` の `padding_mask` を作ります。
8. 画像を `0.0..1.0` に正規化します。
9. `normalize_mean` / `normalize_std` でnormalizeします。
10. HWCからCHWへtransposeします。
11. `pixel_values` は `[1, 3, H, W]` の `np.float32` として返します。
12. `padding_mask` は `[1, H, W]` として返します。

PixAIと違い、OppaiOracleは縦横比を維持してpaddingします。
また、ONNX入力に `padding_mask` がある場合は、そのmaskも渡します。

## target_sizeの決定

`_target_size_from_session(session, preprocess)` は、ONNX入力のうち名前が `pixel_values` のものを探します。
そのshape内の整数次元を集め、末尾2つの最大値を `target_size` とします。

`pixel_values` 入力が見つからない、またはshapeからサイズを取れない場合は、`preprocess["image_size"]` を使います。
それもなければ既定値 `448` です。

## 推論

`get_oppai_oracle_tags(...)` は次の順で動きます。

1. `load_oppai_oracle_preprocess(...)` で前処理設定を取得します。
2. `_target_size_from_session(...)` で入力サイズを決めます。
3. `prepare_image_for_oppai_oracle(...)` で `pixel_values` と `padding_mask` を作ります。
4. ONNX入力名を見て、存在する入力だけ `input_map` に追加します。
5. すべての出力名を取得します。
6. `session.run(...)` を実行します。
7. 既定では最初の出力を予測配列として使います。
8. 出力名に `probabilities` が含まれる場合は、その出力を優先して使います。
9. `postprocess_oppai_oracle(...)` へ渡します。

入力名に基づいて `pixel_values` と `padding_mask` を渡すため、ONNX側が `padding_mask` を持たない場合でも実行できる構造です。

## 後処理

`postprocess_oppai_oracle(...)` は、予測配列とCSVのindex対応からグループ別辞書を作ります。

採用条件は次の通りです。

- `index < len(pred)`
- タグ名が空でない
- タグ名が `<PAD>` でも `<UNK>` でもない
- `float(pred[index]) > threshold`

返り値の主なキーは次の通りです。

- `general`: generalタグ
- `character`: characterタグ
- `rating`: ratingタグ
- `tag`: rating以外をまとめたタグ
- `prediction`: `np.float32` の予測配列

`drop_overlap=True` の場合は、`general` グループに `drop_overlap_tags(...)` が適用されます。

## しきい値

OppaiOracle単体の後処理では、すべてのグループに同じ `threshold` を使います。
既定値は `0.753` です。

pipeline経由では `predict_oppai_oracle_tags(...)` が呼ばれ、OppaiOracle側では候補タグとスコアを返します。
その後、`postprocess.categorize_tags(...)` で `threshold`、`rating_threshold`、`character_threshold`、`copyright_threshold`、`meta_threshold` がカテゴリ別に適用されます。

`oppai-oracle-v1.1` のモデル定義では、general側とcharacter側の推奨しきい値はいずれも `0.753` です。

## pipelineでの扱い

`pipeline._raw_output_for_image(...)` はOppaiOracle familyを次のように呼びます。

- `predict_oppai_oracle_tags(...)`
- `csv_path=model_paths["selected_tags.csv"]`
- `preprocess_path=model_paths.get("preprocessing.json")`

OppaiOracle単体後処理ではcopyright/metaカテゴリが明示されないため、pipeline側の補完判定が重要です。
`tag_catalog.py` は、明示カテゴリに入っていないタグに対して次の順でカテゴリを決めます。

1. `rating:` prefixなら `rating`
2. characterタグ一覧にあれば `character`
3. copyrightタグ一覧にあれば `copyright`
4. metaタグ一覧にあれば `meta`
5. それ以外は `general`

この補完により、OppaiOracleの出力にcopyright/metaタグが混ざっていても、タグ一覧に存在すれば各カテゴリのthresholdとinclude設定の対象になります。

## 実装上の注意点

OppaiOracle familyを変更する場合は次の点に注意してください。

- repository内のファイルは `V1.1_onnx/` 配下ですが、ローカルでは `モデル名.ext` 形式で保存されます。
- `preprocessing.json` は必須扱いですが、読み込み失敗時のfallbackも実装されています。
- 入力はCHW形式です。
- 縦横比を維持したpaddingと `padding_mask` が推論結果に影響します。
- `probabilities` 出力が存在する場合は、それを優先します。
- `<PAD>` と `<UNK>` は出力タグから除外されます。
- `pr_thresholds.json` は現時点では推論後処理に使われません。
