# KeemenaSubwords.jl


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mantzaris.github.io/KeemenaSubwords.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mantzaris.github.io/KeemenaSubwords.jl/dev/)
[![Build Status](https://github.com/mantzaris/KeemenaSubwords.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mantzaris/KeemenaSubwords.jl/actions/workflows/CI.yml?query=branch%3Amain)


---
Subword tokenization methods (BPE, SentencePiece, WordPiece, Unigram) in Julia for text ('keemena') preprocessing pipelines.


Downstream of [KeemenaPreprocessing.jl](https://github.com/mantzaris/KeemenaPreprocessing.jl).
Implements subword tokenization (BPE, WordPiece, SentencePiece, Unigram) for use with Keemena* packages.

## Installation
```julia
] add https://github.com/mantzaris/KeemenaSubwords.jl
```