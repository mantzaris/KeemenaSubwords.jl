function _assert_sentencepiece_unknown_offsets(
    tokenizer::SentencePieceTokenizer,
    clean_text::String,
    unknown_char::String,
)::Nothing
    tokenization_text = tokenization_view(tokenizer, clean_text)
    result = encode_result(
        tokenizer,
        tokenization_text;
        assume_normalized=true,
        return_offsets=true,
        return_masks=true,
        add_special_tokens=false,
    )

    @test result.offsets !== nothing
    offsets = result.offsets

    @test_nowarn assert_offsets_contract(
        tokenization_text,
        offsets;
        require_string_boundaries=true,
    )
    @test offsets_are_nonoverlapping(offsets)

    unk = unk_id(tokenizer)
    unk_indices = findall(id -> id == unk, result.ids)
    @test !isempty(unk_indices)

    for idx in unk_indices
        offset = offsets[idx]
        @test offset != offsets_sentinel()
        @test has_nonempty_span(offset)

        span = try_span_substring(tokenization_text, offset)
        @test span !== nothing
        @test occursin(unknown_char, span)
    end

    return nothing
end

@testset "SentencePiece Unigram unknown token offsets are spanful" begin
    corpus = [
        "hello world",
        "plain ascii corpus",
        "token offsets remain stable",
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
        prune_fraction=0.2,
        model_name="training_unknown_offsets_sentencepiece_unigram",
    )

    _assert_sentencepiece_unknown_offsets(tokenizer, "hello Ω", "Ω")
end

@testset "SentencePiece BPE unknown token offsets are spanful" begin
    corpus = [
        "hello world",
        "plain ascii corpus",
        "token offsets remain stable",
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
        model_name="training_unknown_offsets_sentencepiece_bpe",
    )

    _assert_sentencepiece_unknown_offsets(tokenizer, "hello Ω", "Ω")
end
