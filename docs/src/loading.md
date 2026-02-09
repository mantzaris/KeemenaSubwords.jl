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
xlmr = load_tokenizer(:xlm_roberta_base_sentencepiece_bpe)

# directory/file auto-detection
load_tokenizer("/path/to/model_dir")
load_tokenizer("/path/to/spm.model")
load_tokenizer("/path/to/encoding.tiktoken")

# explicit BPE/ByteBPE paths
load_tokenizer(("/path/to/vocab.txt", "/path/to/merges.txt"))
load_tokenizer(("/path/to/vocab.txt", "/path/to/merges.txt"); format=:bytebpe)

# GPT-2 style byte-level BPE
load_tokenizer("/path/to/gpt2_dir"; format=:bpe_gpt2)
# also supports encoder.json + vocab.bpe filenames

# named spec
load_tokenizer((format=:bpe_gpt2, vocab="/path/to/vocab.txt", merges="/path/to/merges.txt"))
```
