# LLM Cookbook

Practical recipes for model selection, installation, and interop workflows.

Quick links:

- [Structured Outputs and Batching](structured_outputs_and_batching.md)
- [Offsets Alignment Examples](offset_alignment_examples.md)
- [Tokenizer Formats and Required Files](formats.md)
- [Installable Gated Models](gated_models.md)

## Recipe 1: Pick recommended defaults and prefetch

```julia
using KeemenaSubwords

keys = recommended_defaults_for_llms()
prefetch_models(keys)

for key in keys
    println(key, " => ", describe_model(key).format)
end
```

Use this as the first step when you want a curated, ready-to-load starting set.

## Recipe 2: Load and encode with one recommended model

```julia
using KeemenaSubwords

key = first(recommended_defaults_for_llms())
tokenizer = load_tokenizer(key)

tokenization_text = tokenization_view(tokenizer, "hello world")
result = encode_result(
    tokenizer,
    tokenization_text;
    assume_normalized=true,
    return_offsets=true,
    return_masks=true,
    add_special_tokens=true,
)

(key=key, ids=result.ids, tokens=result.tokens)
```

For training-ready batch tensors and padding, continue with
[Structured Outputs and Batching](structured_outputs_and_batching.md).

## Recipe 3: Gated install workflow (LLaMA)

```julia
using KeemenaSubwords

install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])
llama = load_tokenizer(:llama3_8b_tokenizer)
```

You must have accepted upstream license terms and have valid model access.

## Recipe 4: Manual local-path loading for LLaMA tokenizers

```julia
using KeemenaSubwords

# LLaMA2-style SentencePiece
llama2 = load_tokenizer("/path/to/tokenizer.model"; format=:sentencepiece_model)

# LLaMA3-style tokenizer.model with tiktoken text
llama3 = load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)
```

## Recipe 5: Export tokenizer.json for Python Fast tokenizers

```julia
using KeemenaSubwords

tokenizer = load_tokenizer(:core_wordpiece_en)
export_tokenizer(tokenizer, "out_tokenizer"; format=:hf_tokenizer_json)
```

```python
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file="out_tokenizer/tokenizer.json")
```
