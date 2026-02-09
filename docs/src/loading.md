# Loading Tokenizers

```julia
using KeemenaSubwords

# built-in models
bpe = load_tokenizer(:core_bpe_en)
wp = load_tokenizer(:core_wordpiece_en)
sp = load_tokenizer(:core_sentencepiece_unigram_en)
tiktoken = load_tokenizer(:tiktoken_cl100k_base)
gpt2 = load_tokenizer(:openai_gpt2_bpe)
mistral_v1 = load_tokenizer(:mistral_v1_sentencepiece)
phi2 = load_tokenizer(:phi2_bpe)
qwen = load_tokenizer(:qwen2_5_bpe)
bert_multi = load_tokenizer(:bert_base_multilingual_cased_wordpiece)
xlmr = load_tokenizer(:xlm_roberta_base_sentencepiece_bpe)

# directory/file auto-detection
load_tokenizer("/path/to/model_dir")
load_tokenizer("/path/to/spm.model")
load_tokenizer("/path/to/encoding.tiktoken")
load_tokenizer("/path/to/tokenizer.json"; format=:hf_tokenizer_json)

# explicit BPE/ByteBPE paths
load_tokenizer(("/path/to/vocab.txt", "/path/to/merges.txt"))
load_tokenizer(("/path/to/vocab.txt", "/path/to/merges.txt"); format=:bytebpe)

# GPT-2 style byte-level BPE
load_tokenizer("/path/to/gpt2_dir"; format=:bpe_gpt2)
# also supports encoder.json + vocab.bpe filenames

# named spec
load_tokenizer((format=:bpe_gpt2, vocab="/path/to/vocab.txt", merges="/path/to/merges.txt"))

# external user-supplied models (not shipped as built-ins)
load_tokenizer("/path/to/tokenizer.model")                                  # Llama 2 style SentencePiece
load_tokenizer("/path/to/llama3_tokenizer.tiktoken"; format=:tiktoken)      # Llama 3 style tiktoken
load_tokenizer("/path/to/tekken_like.tiktoken"; format=:tiktoken)           # Mistral Tekken-like file

# installable gated models (requires accepted license + token)
install_model!(:llama2_tokenizer; token=ENV["HF_TOKEN"])
install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])
load_tokenizer(:llama3_8b_tokenizer)

# optional key registration for external assets
register_local_model!(
    :my_llama_external,
    "/path/to/tokenizer.model";
    format=:sentencepiece_model,
    family=:llama,
    description="User-supplied Llama tokenizer",
)
load_tokenizer(:my_llama_external)

# optional authenticated Hugging Face download helper
download_hf_files(
    "meta-llama/Llama-3.1-8B",
    ["tokenizer.model"];
    revision="main",
    token=get(ENV, "HF_TOKEN", nothing),
)
```

LLaMA tokenizer files are gated by upstream license terms. KeemenaSubwords does not redistribute those assets and only installs them into your local cache when you call `install_model!`.
