@testset "Unigram training offsets contract" begin
    corpus = [
        "offset checks stay stable",
        "unicode café tokens",
        "emoji 😀 offsets",
    ]

    tokenizer = train_unigram(
        corpus;
        vocab_size=96,
        seed_size=1000,
        num_iters=3,
        max_subword_length=8,
        prune_fraction=0.2,
        special_tokens=Dict(
            :unk => "<UNK>",
            :pad => "<PAD>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        whitespace_marker="",
        model_name="training_offsets_unigram",
    )

    clean_text = "unicode café tokens"
    tokenization_text = tokenization_view(tokenizer, clean_text)
    result = encode_result(
        tokenizer,
        tokenization_text;
        assume_normalized=true,
        return_offsets=true,
        return_masks=true,
        add_special_tokens=true,
    )

    @test result.offsets !== nothing
    offsets = result.offsets

    @test_nowarn assert_offsets_contract(
        tokenization_text,
        offsets;
        require_string_boundaries=true,
    )
    @test offsets_are_nonoverlapping(offsets)

    bos = bos_id(tokenizer)
    eos = eos_id(tokenizer)
    @test bos !== nothing
    @test eos !== nothing
    @test first(result.ids) == bos
    @test last(result.ids) == eos
    @test first(offsets) == offsets_sentinel()
    @test last(offsets) == offsets_sentinel()
end

@testset "Unigram offsets with whitespace marker" begin
    corpus = [
        "marker offsets remain stable",
        "marker based unigram",
    ]

    tokenizer = train_unigram(
        corpus;
        vocab_size=96,
        seed_size=1000,
        num_iters=3,
        max_subword_length=8,
        prune_fraction=0.2,
        special_tokens=Dict(
            :unk => "<UNK>",
            :pad => "<PAD>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        whitespace_marker="▁",
        model_name="training_offsets_unigram_marker",
    )

    clean_text = "marker offsets remain stable"
    tokenization_text = tokenization_view(tokenizer, clean_text)
    result = encode_result(
        tokenizer,
        tokenization_text;
        assume_normalized=true,
        return_offsets=true,
        return_masks=true,
        add_special_tokens=true,
    )

    @test result.offsets !== nothing
    offsets = result.offsets

    @test_nowarn assert_offsets_contract(
        tokenization_text,
        offsets;
        require_string_boundaries=true,
    )
    @test offsets_are_nonoverlapping(offsets)

    bos = bos_id(tokenizer)
    eos = eos_id(tokenizer)
    @test bos !== nothing
    @test eos !== nothing
    @test first(offsets) == offsets_sentinel()
    @test last(offsets) == offsets_sentinel()
end

@testset "Unigram unknown fallback offsets contract" begin
    corpus = [
        "alpha beta",
        "gamma delta",
        "ascii only corpus",
    ]

    tokenizer = train_unigram(
        corpus;
        vocab_size=96,
        seed_size=1000,
        num_iters=3,
        max_subword_length=8,
        prune_fraction=0.2,
        special_tokens=Dict(
            :unk => "<UNK>",
            :pad => "<PAD>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        whitespace_marker="▁",
        model_name="training_offsets_unigram_unknown",
    )

    clean_text = "alpha Ω"
    tokenization_text = tokenization_view(tokenizer, clean_text)
    result = encode_result(
        tokenizer,
        tokenization_text;
        assume_normalized=true,
        return_offsets=true,
        return_masks=true,
        add_special_tokens=true,
    )

    @test any(id -> id == unk_id(tokenizer), result.ids)
    @test result.offsets !== nothing
    offsets = result.offsets

    @test_nowarn assert_offsets_contract(
        tokenization_text,
        offsets;
        require_string_boundaries=true,
    )
    @test offsets_are_nonoverlapping(offsets)

    bos = bos_id(tokenizer)
    eos = eos_id(tokenizer)
    @test bos !== nothing
    @test eos !== nothing
    @test first(result.ids) == bos
    @test last(result.ids) == eos
    @test first(offsets) == offsets_sentinel()
    @test last(offsets) == offsets_sentinel()
end
