# Loading Tokenizers

```julia
using KeemenaSubwords

# built-in models
bpe = load_tokenizer(:core_bpe_en)
wp = load_tokenizer(:core_wordpiece_en)
sp = load_tokenizer(:core_sentencepiece_unigram_en)

# directory/file auto-detection
load_tokenizer("/path/to/model_dir")
load_tokenizer("/path/to/spm.model")

# explicit BPE/ByteBPE paths
load_tokenizer(("/path/to/vocab.txt", "/path/to/merges.txt"))
load_tokenizer(("/path/to/vocab.txt", "/path/to/merges.txt"); format=:bytebpe)

# GPT-2 style byte-level BPE
load_tokenizer("/path/to/gpt2_dir"; format=:bpe_gpt2)

# named spec
load_tokenizer((format=:bpe_gpt2, vocab="/path/to/vocab.txt", merges="/path/to/merges.txt"))
```
