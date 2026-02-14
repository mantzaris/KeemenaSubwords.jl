@testset "SentencePiece Unigram training end-to-end" begin
    corpus = [
        "hello world",
        "sentencepiece unigram training",
        "unicode café tokens",
        "punctuation, offsets!",
    ]

    config_kwargs = (
        vocab_size=96,
        model_type=:unigram,
        special_tokens=Dict(
            :unk => "<unk>",
            :pad => "<pad>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        whitespace_marker="▁",
        seed_size=800,
        num_iters=2,
        max_subword_length=8,
        prune_fraction=0.2,
        model_name="training_e2e_sentencepiece_unigram",
        version=v"0.3.0",
    )

    training = train_sentencepiece_result(corpus; config_kwargs...)
    tokenizer = training.tokenizer
    @test tokenizer isa SentencePieceTokenizer
    @test tokenizer.inner isa UnigramTokenizer
    @test training.config.model_type == :unigram
    @test training.artifacts.model_type == :unigram
    @test training.artifacts.whitespace_marker == "▁"
    @test training.artifacts.inner_artifacts isa KeemenaSubwords.Training.UnigramTrainingArtifacts

    samples = [
        "hello world",
        "unicode café tokens",
    ]

    for text in samples
        plain_ids = encode(tokenizer, text; add_special_tokens=false)
        @test tokenize(tokenizer, text) == String[id_to_token(tokenizer, id) for id in plain_ids]
        ids = encode(tokenizer, text; add_special_tokens=true)
        @test decode(tokenizer, ids) == text
    end

    outdir = mktempdir()
    save_tokenizer(tokenizer, outdir)
    reloaded = load_tokenizer(outdir; format=:auto)
    @test reloaded isa SentencePieceTokenizer

    for text in samples
        @test tokenize(reloaded, text) == tokenize(tokenizer, text)
        @test encode(reloaded, text; add_special_tokens=true) == encode(tokenizer, text; add_special_tokens=true)
        @test decode(reloaded, encode(reloaded, text; add_special_tokens=true)) ==
              decode(tokenizer, encode(tokenizer, text; add_special_tokens=true))
    end
end

@testset "SentencePiece BPE training end-to-end" begin
    corpus = [
        "hello world",
        "sentencepiece bpe training",
        "unicode café tokens",
        "punctuation, offsets!",
    ]

    config_kwargs = (
        vocab_size=96,
        model_type=:bpe,
        min_frequency=1,
        special_tokens=Dict(
            :unk => "<unk>",
            :pad => "<pad>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        whitespace_marker="▁",
        model_name="training_e2e_sentencepiece_bpe",
        version=v"0.3.0",
    )

    training = train_sentencepiece_result(corpus; config_kwargs...)
    tokenizer = training.tokenizer
    @test tokenizer isa SentencePieceTokenizer
    @test tokenizer.inner isa BPETokenizer
    @test tokenizer.inner.end_of_word_marker === nothing
    @test training.config.model_type == :bpe
    @test training.artifacts.model_type == :bpe
    @test training.artifacts.inner_artifacts isa KeemenaSubwords.Training.BPETrainingArtifacts

    samples = [
        "hello world",
        "unicode café tokens",
    ]

    for text in samples
        plain_ids = encode(tokenizer, text; add_special_tokens=false)
        @test tokenize(tokenizer, text) == String[id_to_token(tokenizer, id) for id in plain_ids]
        ids = encode(tokenizer, text; add_special_tokens=true)
        @test decode(tokenizer, ids) == text
    end

    outdir = mktempdir()
    save_tokenizer(tokenizer, outdir)
    reloaded = load_tokenizer(outdir; format=:sentencepiece_model)
    @test reloaded isa SentencePieceTokenizer

    for text in samples
        @test tokenize(reloaded, text) == tokenize(tokenizer, text)
        @test encode(reloaded, text; add_special_tokens=true) == encode(tokenizer, text; add_special_tokens=true)
        @test decode(reloaded, encode(reloaded, text; add_special_tokens=true)) ==
              decode(tokenizer, encode(tokenizer, text; add_special_tokens=true))
    end

    second = train_sentencepiece_result(corpus; config_kwargs...)
    second_inner = second.artifacts.inner_artifacts
    first_inner = training.artifacts.inner_artifacts
    @test second_inner.vocab_tokens == first_inner.vocab_tokens
    @test second_inner.merge_pairs == first_inner.merge_pairs
    @test second_inner.pair_ranks == first_inner.pair_ranks

    reversed_training = train_sentencepiece_result(reverse(corpus); config_kwargs...)
    reversed_inner = reversed_training.artifacts.inner_artifacts
    @test reversed_inner.vocab_tokens == first_inner.vocab_tokens
    @test reversed_inner.merge_pairs == first_inner.merge_pairs
    @test reversed_inner.pair_ranks == first_inner.pair_ranks
end
