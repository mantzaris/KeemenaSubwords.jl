function _assert_sentencepiece_offsets(
    tokenizer::SentencePieceTokenizer,
    clean_text::String,
)::Nothing
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

    special_ids = Set(values(special_tokens(tokenizer)))
    unk_token = id_to_token(tokenizer, unk_id(tokenizer))
    sentinel = offsets_sentinel()

    for (token, id, offset, mask) in zip(result.tokens, result.ids, offsets, masks)
        if offset == sentinel
            @test id in special_ids
            @test mask == 1
            continue
        end

        has_nonempty_span(offset) || continue
        span = try_span_substring(tokenization_text, offset)
        @test span !== nothing

        id in special_ids && continue
        token == unk_token && continue

        expected = replace(token, tokenizer.whitespace_marker => "")
        @test span == expected
    end

    bos = bos_id(tokenizer)
    eos = eos_id(tokenizer)
    if bos !== nothing
        @test first(result.ids) == bos
        @test first(offsets) == sentinel
        @test first(masks) == 1
    end
    if eos !== nothing
        @test last(result.ids) == eos
        @test last(offsets) == sentinel
        @test last(masks) == 1
    end

    return nothing
end

@testset "SentencePiece Unigram training offsets contract" begin
    corpus = [
        "offset checks stay stable",
        "unicode café tokens",
        "emoji 😀 offsets",
    ]

    tokenizer = train_sentencepiece(
        corpus;
        vocab_size=96,
        model_type=:unigram,
        special_tokens=Dict(
            :unk => "<unk>",
            :pad => "<pad>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        whitespace_marker="▁",
        seed_size=1000,
        num_iters=3,
        max_subword_length=8,
        model_name="training_offsets_sentencepiece_unigram",
    )

    _assert_sentencepiece_offsets(tokenizer, "unicode café tokens")
end

@testset "SentencePiece BPE training offsets contract" begin
    corpus = [
        "offset checks stay stable",
        "unicode café tokens",
        "punctuation, offsets!",
    ]

    tokenizer = train_sentencepiece(
        corpus;
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
        model_name="training_offsets_sentencepiece_bpe",
    )

    _assert_sentencepiece_offsets(tokenizer, "punctuation, offsets!")

    paragraph = "offset checks stay stable, even with punctuation and café tokens."
    _assert_sentencepiece_offsets(tokenizer, paragraph)
end
