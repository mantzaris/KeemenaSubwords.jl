@testset "Unigram unknown token offsets are spanful" begin
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
        model_name="training_unknown_offsets_unigram",
    )

    clean_text = "alpha Ω"
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
        @test occursin("Ω", span)
    end
end
