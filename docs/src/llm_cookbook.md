# LLM Cookbook

## OpenAI tiktoken encodings

```julia
prefetch_models([:tiktoken_cl100k_base, :tiktoken_o200k_base])
tt = load_tokenizer(:tiktoken_cl100k_base)
encode(tt, "hello world")
```

## Mistral SentencePiece

```julia
prefetch_models([:mistral_v3_sentencepiece])
mistral = load_tokenizer(:mistral_v3_sentencepiece)
```

## Qwen tokenizer.json-first loading

```julia
prefetch_models([:qwen2_5_bpe])
qwen = load_tokenizer(:qwen2_5_bpe)
```

## LLaMA workflow A: install (gated)

```julia
install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])
llama = load_tokenizer(:llama3_8b_tokenizer)
```

## LLaMA workflow B: manual local path

```julia
# SentencePiece-style
llama2 = load_tokenizer("/path/to/tokenizer.model"; format=:sentencepiece_model)

# LLaMA3-style tokenizer.model with tiktoken text
llama3 = load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)
```

## Pick practical defaults

```julia
for key in recommended_defaults_for_llms()
    println(key)
end
```
