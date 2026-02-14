@testset "SentencePiece longform smoke" begin
    text = "SentencePiece training should stay stable, even on longer text. The café example checks multibyte behavior, and offsets should remain aligned."

    corpus = [
        "this package trains sentencepiece tokenizers",
        "offsets should remain deterministic and stable",
        "unicode words like café should roundtrip",
        "punctuation and commas are part of the test corpus",
        text,
    ]

    tokenizer = train_sentencepiece(
        corpus;
        vocab_size=192,
        model_type=:unigram,
        special_tokens=Dict(
            :unk => "<unk>",
            :pad => "<pad>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        whitespace_marker="▁",
        seed_size=1600,
        num_iters=4,
        max_subword_length=10,
        prune_fraction=0.2,
        model_name="training_longform_smoke_sentencepiece",
    )

    ids = encode(tokenizer, text; add_special_tokens=true)
    @test decode(tokenizer, ids) == text

    tokenization_text = tokenization_view(tokenizer, text)
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
    @test offsets_are_nonoverlapping(
        offsets;
        ignore_sentinel=true,
        ignore_empty=true,
    )
end
