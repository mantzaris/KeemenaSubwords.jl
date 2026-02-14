@testset "WordPiece training offsets contract" begin
    corpus = [
        "hello world offsets",
        "unicode café offsets",
        "wordpiece offsets stay stable",
    ]

    tokenizer = train_wordpiece(
        corpus;
        vocab_size=128,
        min_frequency=1,
        special_tokens=Dict(
            :unk => "[UNK]",
            :pad => "[PAD]",
            :cls => "[CLS]",
            :sep => "[SEP]",
            :mask => "[MASK]",
        ),
        continuation_prefix="##",
        model_name="training_offsets_wordpiece",
    )

    clean_text = "unicode café offsets"
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
    @test result.special_tokens_mask !== nothing
    offsets = result.offsets
    masks = result.special_tokens_mask

    @test length(result.ids) == length(result.tokens)
    @test length(result.tokens) == length(offsets)
    @test length(result.tokens) == length(masks)

    @test_nowarn assert_offsets_contract(
        tokenization_text,
        offsets;
        require_string_boundaries=true,
    )
    @test offsets_are_nonoverlapping(offsets)

    cls = get(special_tokens(tokenizer), :cls, nothing)
    sep = get(special_tokens(tokenizer), :sep, nothing)
    sentinel = offsets_sentinel()
    if cls !== nothing
        @test first(result.ids) == cls
        @test first(offsets) == sentinel
        @test first(masks) == 1
    end
    if sep !== nothing
        @test last(result.ids) == sep
        @test last(offsets) == sentinel
        @test last(masks) == 1
    end

    for (token, offset) in zip(result.tokens, offsets)
        has_nonempty_span(offset) || continue
        span = try_span_substring(tokenization_text, offset)
        @test span !== nothing
        @test span == replace(token, tokenizer.continuation_prefix => "")
    end
end
