# comfy-tagger

ComfyUI用の画像taggerノードです。WD系tagger、PixAI tagger、OppaiOracle V1.1で画像ごとのタグ文字列を推定します。

## Installation

`custom_nodes` に配置し、依存関係をインストールしてください。

```bash
python -m pip install -r custom_nodes/comfy-tagger/requirements.txt
```

## Nodes

### Tagger

画像ごとにtaggerモデルでタグを推定し、`list[str]` として返します。

対応モデル:

- `wd-eva02-large-tagger-v3`
- `wd-vit-large-tagger-v3`
- `wd-v1-4-swinv2-tagger-v2`
- `wd-vit-tagger-v3`
- `pixai-tagger-v0.9`
- `oppai-oracle-v1.1`

Inputs:

- `images`
- `model`
- `use_default_threshold`
- `threshold`
- `include_character`
- `character_threshold`
- `include_copyright`
- `copyright_threshold`
- `include_meta`
- `meta_threshold`
- `include_rating`
- `rating_threshold`
- `replace_underscore`
- `drop_overlap`
- `drop_blacklist`
- `drop_basic_character`
- `sort_mode`
- `prioritize_people_tags`
- `trailing_comma`
- `exclude_tags`

`use_default_threshold` が有効な場合、`threshold` / `character_threshold` / `copyright_threshold` / `meta_threshold` / `rating_threshold` の入力値ではなく、モデル定義側の推奨値を使います。characterは `default_character_threshold -> default_threshold`、copyright/ratingは `カテゴリ別threshold -> default_character_threshold -> default_threshold`、metaは `default_meta_threshold -> default_threshold` の優先順で決まります。`oppai-oracle-v1.1` はV1.1 ONNX版を使用し、既定の `threshold` は `0.753` です。

`character` / `copyright` / `meta` タグ一覧は初回使用時に `jupo-ai/danbooru-characters` から自動取得します。モデル出力が `general` / `character` / `copyright` / `meta` / `rating` に分かれている場合は各カテゴリのしきい値を使い、分かれていない場合はタグ一覧と `rating:` prefix でカテゴリを判定します。

Taggerモデルは初回実行時に自動ダウンロードされます。

保存先は、ComfyUIの `extra_model_paths.yaml` で `tagger` が登録されている場合はそのディレクトリを使います。登録されていない場合は、この拡張直下の `tagger_models` を使用します。

例:

```yaml
my_models:
  base_path: /path/to/models
  tagger: tagger
```

## License

MIT License. See [LICENSE](LICENSE).

## Credits

この拡張のtagger実装には、[deepghs/imgutils](https://github.com/deepghs/imgutils) のコードを参考・利用している部分があります。
