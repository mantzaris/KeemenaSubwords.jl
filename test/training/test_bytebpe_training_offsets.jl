@testset "Byte-level BPE training offsets contract" begin
    corpus = [
        "byte offsets stay deterministic",
        "café 😀 offsets",
        "inserted specials use sentinel spans",
    ]

    tokenizer = train_bytebpe(
        corpus;
        vocab_size=320,
        min_frequency=1,
        special_tokens=Dict(
            :unk => "<UNK>",
            :pad => "<PAD>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        end_of_word_marker="</w>",
        include_full_byte_alphabet=true,
        model_name="training_offsets_bytebpe",
    )

    clean_text = "café 😀 offsets"
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

    @test_nowarn assert_offsets_contract(
        tokenization_text,
        offsets;
        require_string_boundaries=false,
    )
    @test offsets_are_nonoverlapping(offsets)

    nonempty = [offset for offset in offsets if has_nonempty_span(offset)]
    @test !isempty(nonempty)
    @test any(offset -> !isempty(span_codeunits(tokenization_text, offset)), nonempty)

    bos = bos_id(tokenizer)
    eos = eos_id(tokenizer)
    @test bos !== nothing
    @test eos !== nothing
    @test first(result.ids) == bos
    @test last(result.ids) == eos
    @test first(offsets) == offsets_sentinel()
    @test last(offsets) == offsets_sentinel()
end
