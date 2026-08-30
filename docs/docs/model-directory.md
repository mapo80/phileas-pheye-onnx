# Model Directory

Local inference loads a model from a directory on disk. The `modelPath` you set on a PhEye filter
(see [Configuration](configuration.md)) points at that directory.

The directory's contents also decide **which** detector is built. The two layouts are disjoint, so
this is a fact about the directory rather than a guess, and a directory that matches neither — or
somehow both — is refused at construction instead of being half-loaded.

## GLiNER

| File | Purpose |
|---|---|
| `model.onnx` (or `onnx/model.onnx`) | The exported GLiNER model. |
| `tokenizer.json` | The HuggingFace fast tokenizer. |
| `gliner_config.json` | Span width, maximum length, and prompt tokens. |

This is the layout produced by `ph-eye-model-training`'s `./hub/publish.sh`. The published PhEye PII
name models on the Hugging Face Hub follow it — `philterd/ph-eye-pii-en-small`,
`philterd/ph-eye-pii-en-medium`, `philterd/ph-eye-pii-en-large`. Download one into a local directory
and point `modelPath` at it.

`gliner_config.json` drives the pipeline: maximum span width, maximum sequence length, and the
prompt tokens used to build the model input. See [How It Works](how-it-works.md).

## Token classification

| File | Purpose |
|---|---|
| `model.onnx` (or `onnx/model.onnx`) | The exported model: `input_ids`, `attention_mask` → `logits`. |
| `tokenizer.json` | The HuggingFace fast tokenizer. |
| `config.json` | The HuggingFace model config. `id2label` is the taxonomy, in BIO notation. |
| `token_classification_config.json` | The inference window, and optionally the calibrated threshold. |

`config.json` is read for `id2label` only, and it is authoritative: it is the taxonomy the weights
were trained on. Labels are ordered by class index rather than by their order in the file — JSON
objects have none, and a label read at the wrong index mislabels every document.

`token_classification_config.json` carries what the HuggingFace config does not:

```json
{
  "max_words": 120,
  "overlap_words": 20,
  "max_tokens": 8192,
  "words_splitter_type": "whitespace",
  "calibrated_threshold": 0.92
}
```

- `max_words` / `overlap_words`: the inference window, in whitespace-delimited words. **Not
  defaulted.** An encoder's `config.json` advertises its positional limit
  (`max_position_embeddings`), which is a ceiling, not the length the model is meant to be run at;
  running a model at the wrong window degrades it quietly.
- `max_tokens`: the encoder's hard sub-token capacity, past which the graph cannot run. It is a
  safety limit, not a quality knob — see the sub-token fallback in
  [Token-Classification Models](token-classification.md#long-input).
- `calibrated_threshold` *(optional)*: the decode threshold this model was calibrated at. It
  replaces the library default when the caller has set no threshold of its own, and never overrides
  one the caller did set.

`scripts/package_token_classification_model.py` builds this layout from a HuggingFace checkpoint and
an exported graph.

## Fail-closed

A missing or unreadable file, a malformed config, an `id2label` that is not BIO, a `max_words` or
`max_tokens` that is not positive, an `overlap_words` at least as wide as the window, a
`words_splitter_type` other than `whitespace`, a graph whose input signature is not the expected
one, or a class count that disagrees with `id2label` — each is a constructor failure.

That strictness is the point. For a redaction component the dangerous failure is not an exception:
it is a detector that builds successfully and then finds nothing, because a document with no
personal data looks exactly like a document that was never examined.
