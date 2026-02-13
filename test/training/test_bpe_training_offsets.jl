@testset "BPE training offsets contract" begin
    corpus = [
        "offset checks stay stable",
        "stable offsets remain deterministic",
        "bos eos tokens are special",
    ]

    tokenizer = train_bpe(
        corpus;
        vocab_size=96,
        min_frequency=1,
        special_tokens=Dict(
            :unk => "<UNK>",
            :pad => "<PAD>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        end_of_word_marker="</w>",
        model_name="training_offsets_bpe",
    )

    clean_text = "offset checks stay stable"
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

    @test result.metadata.assume_normalized
    @test result.metadata.offsets_reference == :input_text
    @test length(offsets) == length(result.ids)
    @test length(masks) == length(result.ids)

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
    @test first(masks) == 1
    @test last(masks) == 1
end
