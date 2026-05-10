# PixAI family詳細解説

PixAI familyは、`deepghs/pixai-tagger-v0.9-onnx` のONNX taggerを扱う実装です。
実装本体は `py/modules/tagger/pixai.py` にあります。

## 対応モデル

現在のPixAI familyには次のモデルが登録されています。

| モデル名 | repo_id |
| --- | --- |
| `pixai-tagger-v0.9` | `deepghs/pixai-tagger-v0.9-onnx` |

必要ファイルと任意ファイルは次の通りです。

| repository内ファイル | ローカル識別子 | 必須 |
| --- | --- | --- |
| `model.onnx` | `onnx` | yes |
| `selected_tags.csv` | `csv` | yes |
| `thresholds.csv` | `thresholds.csv` | no |
| `preprocess.json` | `preprocess.json` | no |

`ModelSpec` のgetterで取得できる実効しきい値は次の通りです。

- general: `0.30`
- character: `0.85`
- copyright: `0.85`
- meta: `0.30`
- rating: `0.85`

`ModelSpec` では `default_threshold=0.30`、`default_character_threshold=0.85` を設定し、copyright/rating/metaの個別値は省略しています。
そのため、copyright/ratingは `default_character_threshold` へfallbackします。
metaは `default_threshold` へfallbackします。

## ラベル読み込み

`load_pixai_labels(csv_path, no_underline=False)` は `selected_tags.csv` を読み込み、次の3要素を返します。

1. `tag_names`
2. `indexes_by_category`
3. `ips_mapping`

CSVに `id` 列がある場合、`id` 昇順に並べ替えてからindexを割り当てます。
これは、CSVの物理的な行順よりもモデル出力indexとの対応を優先するためです。

カテゴリ判定は次の順で行われます。

1. `category` 列があり、空でなければその整数値を使います。
2. `category` がない場合は `is_char` 列を見ます。
3. `is_char` が `1` / `true` / `yes` ならcategory `4` とします。
4. それ以外はcategory `0` とします。

一般的にはcategory `0` がgeneral、category `4` がcharacterです。

## IP情報

PixAIのCSVには `ips` 列が含まれる場合があります。
`load_pixai_labels(...)` は `ips` をJSONとして読み込み、失敗した場合は `ast.literal_eval(...)` でも解釈を試みます。

読み取れたIP情報は `ips_mapping` に保存されます。
これは、キャラクタータグから作品・シリーズなどのIP候補を集計するために使われます。

`postprocess_pixai(...)` は次の追加キーを返します。

- `ips_mapping`: 採用されたcharacterタグからIP候補への対応
- `ips_count`: IP候補ごとの出現数
- `ips`: 出現数降順、同数なら名前昇順のIP候補リスト

現在の `tag_image(...)` では、これらのIP情報は最終タグ文字列には直接使われません。
ただし、`tag_image_grouped(...)` や将来の拡張で利用できます。

## しきい値ファイル

`load_pixai_thresholds(path)` は `thresholds.csv` を読み込み、カテゴリごとのしきい値とカテゴリ名を返します。

返り値は次の2要素です。

1. `thresholds`: `dict[int, float]`
2. `category_names`: `dict[int, str]`

`thresholds.csv` がない場合は、次のfallbackが使われます。

- category `0`: threshold `0.30`, name `general`
- category `4`: threshold `0.85`, name `character`

`thresholds.csv` が存在する場合も、category `0` と `4` の既定値は不足時に補完されます。

## 前処理

`prepare_image_for_pixai(image_np, target_size, preprocess_path=None)` はPixAI向けに画像を整えます。

処理内容は次の通りです。

1. `np.ndarray` からPillowのRGB画像を作ります。
2. 画像を `target_size x target_size` にBICUBICでresizeします。
3. `0..255` を `0.0..1.0` に正規化します。
4. `preprocess.json` に応じたmean/stdでnormalizeします。
5. HWCからCHWへtransposeします。
6. batch次元を追加して `[1, 3, H, W]` にします。
7. `np.float32` として返します。

WD14と異なり、元画像の縦横比を保つpaddingは行いません。
入力は正方形へ直接resizeされます。

## target_sizeの決定

`_target_size_from_session(session)` はONNX入力shapeから整数次元を集め、末尾2つの最大値を `target_size` として使います。

shapeから十分な整数次元が取得できない場合は `448` にfallbackします。

## normalize設定

`_normalize_from_preprocess(preprocess_path)` は `preprocess.json` を読み、正規化mean/stdを決めます。

処理は単純で、JSON全体を文字列化して小文字にした内容に `imagenet` が含まれるかを見ます。

- `imagenet` を含む場合: ImageNet mean/std
  - mean: `[0.485, 0.456, 0.406]`
  - std: `[0.229, 0.224, 0.225]`
- それ以外または読み込み失敗時:
  - mean: `[0.5, 0.5, 0.5]`
  - std: `[0.5, 0.5, 0.5]`

`preprocess.json` は任意ファイルなので、存在しない場合も後者の設定で実行できます。

## 推論

`get_pixai_tags(...)` は次の順で動きます。

1. ONNX入力情報を取得します。
2. `_target_size_from_session(...)` で入力サイズを決めます。
3. `prepare_image_for_pixai(...)` で画像を前処理します。
4. すべての出力名を取得します。
5. `session.run(...)` で全出力を取得します。
6. 出力名が `prediction` のものを優先して使います。
7. `prediction` がなければ `logits` を探します。
8. `logits` もなければ最初の出力を使います。
9. `prediction` 以外を使った場合は `sigmoid(...)` で確率化します。
10. `postprocess_pixai(...)` へ渡します。

PixAIのONNXは出力名の違いに備えた実装になっています。
すでに確率である `prediction` がある場合はそのまま使い、logit系の出力ではsigmoidを通します。

## 後処理

`postprocess_pixai(...)` はカテゴリごとにタグを選択し、カテゴリ名をキーにした辞書を作ります。

カテゴリごとの処理は次の通りです。

1. `load_pixai_labels(...)` でタグ名とカテゴリindexを取得します。
2. `load_pixai_thresholds(...)` でカテゴリしきい値とカテゴリ名を取得します。
3. 引数 `threshold` が指定されていればcategory `0` のしきい値を上書きします。
4. 引数 `character_threshold` が指定されていればcategory `4` のしきい値を上書きします。
5. 各カテゴリについて、`pred[index] > category_threshold` のタグを採用します。
6. category `0` かつ `drop_overlap=True` の場合は重複タグを除去します。
7. category `9` 以外を `tag` に統合します。
8. characterタグからIP情報を集計します。

返り値の主なキーは次の通りです。

- `general`: generalタグ
- `character`: characterタグ
- その他 `thresholds.csv` に由来するカテゴリ名
- `tag`: category `9` 以外をまとめたタグ
- `ips_mapping`
- `ips_count`
- `ips`
- `prediction`

## pipelineでの扱い

`pipeline._raw_output_for_image(...)` はPixAI familyを次のように呼びます。

- `predict_pixai_tags(...)`
- `csv_path=model_paths["selected_tags.csv"]`
- `thresholds_path=model_paths.get("thresholds.csv")`
- `preprocess_path=model_paths.get("preprocess.json")`

PixAI側ではモデル出力とラベルCSVを対応付けた候補を返します。
最終的なカテゴリ別しきい値、アンダースコア置換、重複除去は `postprocess.categorize_tags(...)` で適用されます。

PixAIの後処理が `general` や `character` という明示カテゴリを返す場合、pipelineはそのカテゴリを優先します。
明示カテゴリに入らず `tag` にだけ含まれるタグは、`tag_catalog.py` のcharacter/copyright/meta/rating判定で補完されます。

## 実装上の注意点

PixAI familyを変更する場合は次の点に注意してください。

- `id` 列があるCSVでは、行順ではなく `id` 昇順がモデル出力indexと対応します。
- `thresholds.csv` は任意なので、存在しない場合のfallbackを壊さないようにします。
- `preprocess.json` も任意なので、読み込み失敗時は安全な既定値で続行します。
- 入力はCHW形式です。
- 現在の前処理は縦横比を維持せず正方形resizeします。
- `prediction` 出力は確率として扱い、`logits` やその他出力はsigmoidで確率化します。
- `ips` 系の戻り値は通常の文字列出力には使われませんが、API利用者にとって有用な補助情報です。
