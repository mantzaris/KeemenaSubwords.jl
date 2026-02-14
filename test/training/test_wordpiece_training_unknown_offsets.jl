@testset "WordPiece unknown token offsets contract" begin
    corpus = [
        "hello world",
        "hello world hello",
    ]

    tokenizer = train_wordpiece(
        corpus;
        vocab_size=96,
        min_frequency=1,
        special_tokens=Dict(
            :unk => "[UNK]",
            :pad => "[PAD]",
            :cls => "[CLS]",
            :sep => "[SEP]",
            :mask => "[MASK]",
        ),
        continuation_prefix="##",
        model_name="training_unknown_offsets_wordpiece",
    )

    clean_text = "hello zzz"
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
    @test length(result.tokens) == length(offsets)

    @test_nowarn assert_offsets_contract(
        tokenization_text,
        offsets;
        require_string_boundaries=true,
    )
    @test offsets_are_nonoverlapping(offsets)

    unk_token = tokenizer.unk_token
    unk_positions = findall(token -> token == unk_token, result.tokens)
    @test length(unk_positions) == 1
    unk_idx = only(unk_positions)
    unk_offset = offsets[unk_idx]
    @test unk_offset != offsets_sentinel()
    @test has_nonempty_span(unk_offset)

    zzz_span = findfirst("zzz", tokenization_text)
    @test zzz_span isa UnitRange{Int}
    zzz_range = zzz_span::UnitRange{Int}
    expected_unk_offset = (first(zzz_range), last(zzz_range) + 1)
    @test unk_offset == expected_unk_offset
    @test try_span_substring(tokenization_text, unk_offset) == "zzz"

    cls = get(special_tokens(tokenizer), :cls, nothing)
    sep = get(special_tokens(tokenizer), :sep, nothing)
    sentinel = offsets_sentinel()
    if cls !== nothing
        @test first(result.ids) == cls
        @test first(offsets) == sentinel
    end
    if sep !== nothing
        @test last(result.ids) == sep
        @test last(offsets) == sentinel
    end
end
