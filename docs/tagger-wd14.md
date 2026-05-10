# WD14 family詳細解説

WD14 familyは、SmilingWolf系のONNX taggerを扱う実装です。
実装本体は `py/modules/tagger/wd14.py` にあります。

## 対応モデル

現在のWD14 familyには次のモデルが登録されています。

| モデル名 | repo_id |
| --- | --- |
| `wd-eva02-large-tagger-v3` | `SmilingWolf/wd-eva02-large-tagger-v3` |
| `wd-vit-large-tagger-v3` | `SmilingWolf/wd-vit-large-tagger-v3` |
| `wd-v1-4-swinv2-tagger-v2` | `SmilingWolf/wd-v1-4-swinv2-tagger-v2` |
| `wd-vit-tagger-v3` | `SmilingWolf/wd-vit-tagger-v3` |

必要ファイルは共通です。

- `model.onnx`
- `selected_tags.csv`

`ModelSpec` のgetterで取得できる実効しきい値は次の通りです。

- general: `0.35`
- character: `0.85`
- copyright: `0.85`
- meta: `0.35`
- rating: `0.85`

`ModelSpec` では `default_threshold=0.35`、`default_character_threshold=0.85` を設定し、copyright/rating/metaの個別値は省略しています。
そのため、copyright/ratingは `default_character_threshold` へfallbackします。
metaは `default_threshold` へfallbackします。

## ラベル読み込み

`load_wd14_labels(csv_path, no_underline=False)` は `selected_tags.csv` を読み込み、タグ名とカテゴリ別indexを返します。

返り値は次の5要素です。

1. `tag_names`
2. `rating_indexes`
3. `general_indexes`
4. `copyright_indexes`
5. `character_indexes`
6. `meta_indexes`

CSVの `category` 値は次のように解釈されます。

| category | グループ |
| --- | --- |
| `9` | `rating` |
| `0` | `general` |
| `3` | `copyright` |
| `4` | `character` |
| `5` | `meta` |

読み込み結果は `_LABEL_CACHE` に保存されます。
キャッシュキーは `(csv_path文字列, no_underline)` です。
同じCSVでもアンダースコア置換の有無でタグ名が変わるため、別キャッシュとして扱われます。

## アンダースコア置換

`no_underline=True` の場合、CSVから読んだタグ名に `format.remove_underline(...)` を適用します。

通常は `red_hair` が `red hair` のように変換されます。
ただし、`0_0` や `>_<` など顔文字系タグは例外として維持されます。

`get_wd14_tags(...)` の `no_underline=True`、またはpipeline APIの `replace_underscore=True` でスペース区切りのタグ名として処理されます。

## 前処理

`prepare_image_for_wd14(image_np, target_size)` はWD系モデル向けに画像を整えます。

処理内容は次の通りです。

1. `np.ndarray` からPillowのRGB画像を作ります。
2. 元画像の縦横の大きい方を基準に正方形キャンバスを作ります。
3. 背景色は白 `(255, 255, 255)` です。
4. 元画像を中央に貼り付けます。
5. 正方形サイズが `target_size` と違う場合はBICUBICでresizeします。
6. `np.float32` 配列へ変換します。
7. RGBをBGR順に反転します。
8. batch次元を追加して `[1, H, W, 3]` にします。

`target_size` はONNX入力のshapeから取得されます。
`get_wd14_tags(...)` では `session.get_inputs()[0].shape[1]` を使います。

WD14 familyは、PixAIやOppaiOracleと違ってCHWではなく、NHWC形式で入力します。

## 推論

`get_wd14_tags(...)` は次の順で動きます。

1. ONNX入力情報を取得します。
2. 入力shapeから `target_size` を決めます。
3. `prepare_image_for_wd14(...)` で画像を前処理します。
4. 最初の出力名を取得します。
5. `session.run(...)` を実行します。
6. 先頭batchの予測配列を `postprocess_wd14(...)` へ渡します。

WD14系モデルの出力は、各タグに対する確率または確率相当のスコア配列として扱われます。

## 後処理

`postprocess_wd14(...)` は、予測配列とCSVのindex対応からカテゴリ別辞書を作ります。

返り値の主なキーは次の通りです。

- `rating`: ratingタグ全件とスコア
- `general`: しきい値を超えたgeneralタグ
- `copyright`: しきい値を超えたcopyrightタグ
- `character`: characterしきい値を超えたcharacterタグ
- `meta`: しきい値を超えたmetaタグ
- `tag`: `character`、`copyright`、`general`、`meta` をまとめた辞書
- `prediction`: `np.float32` の予測配列

`rating` はしきい値で絞らず全件を返します。
ただし、pipeline APIでは後段の `postprocess.categorize_tags(...)` が `include_rating` と `rating_threshold` を適用するため、ratingを含めるかどうかは最終的に統一後処理側で決まります。

## しきい値

WD14 family単体の後処理では次のしきい値を使います。

- `threshold`: general / copyright用
- `character_threshold`: character用

`score > threshold` のタグだけが採用されます。

pipeline経由では `predict_wd14_tags(...)` が呼ばれ、WD14側では全候補を返します。
最終的なしきい値判定は `postprocess.categorize_tags(...)` で行われます。

## MCutしきい値

`mcut_threshold(probs)` は、スコア降順に並べたときの隣接差分が最大になる位置を探し、その2値の中間をしきい値にする関数です。

`postprocess_wd14(...)` には次の引数があります。

- `general_mcut_enabled`
- `character_mcut_enabled`

ただし、現在の `get_wd14_tags(...)` からは有効化されていません。
将来的に自動しきい値を使う場合に利用できる補助実装です。

character MCutでは、極端に低いしきい値を避けるため `max(0.15, mcut値)` が使われます。

## 重複タグ除去

`postprocess_wd14(...)` は `drop_overlap=True` の場合、generalタグに `drop_overlap_tags(...)` を適用します。

pipeline APIでは、重複除去はfamily共通の `postprocess.categorize_tags(...)` 側で実施されます。

## pipelineでの扱い

`pipeline._raw_output_for_image(...)` はWD14 familyを次のように呼びます。

- `predict_wd14_tags(...)`
- `csv_path=model_paths["selected_tags.csv"]`

WD14はCSVにカテゴリ情報が明示されているため、pipeline側では `rating` / `general` / `copyright` / `character` / `meta` の明示カテゴリがそのまま使われます。
その後、include設定、カテゴリ別しきい値、除外フィルタ、並べ替え、文字列化が適用されます。

## 実装上の注意点

WD14 familyを変更する場合は次の点に注意してください。

- `selected_tags.csv` の行順とONNX出力配列のindexは対応している必要があります。
- `category` 値が想定外のタグはカテゴリ別indexへ入りません。
- `rating` は単体後処理では全件返す設計です。
- 入力はBGRのNHWCです。
- 白背景で正方形paddingするため、透過背景や余白の扱いを変えると推論結果が変わる可能性があります。
- キャッシュはCSVパスと `no_underline` ごとに分かれます。
