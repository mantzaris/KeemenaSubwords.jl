@testset "Section 26d HF RobertaProcessing trim_offsets contract" begin
    corpus = [
        "hello world",
        "hello      world",
        "   ",
        "café 🙂",
    ]

    tokenizer = train_hf_roberta_bytebpe(
        corpus;
        vocab_size=320,
        min_frequency=1,
        special_tokens=Dict(
            :unk => "<unk>",
            :pad => "<pad>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        end_of_word_marker="</w>",
        add_prefix_space=true,
        trim_offsets=true,
        use_regex=true,
        nfkc=false,
        lowercase=false,
        model_name="hf_roberta_trim_offsets_contract",
        version=v"0.3.0",
    )

    text = "      "
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
    @test result.special_tokens_mask !== nothing
    offsets = result.offsets
    masks = result.special_tokens_mask
    sentinel = offsets_sentinel()

    @test_nowarn assert_offsets_contract(
        tokenization_text,
        offsets;
        require_string_boundaries=false,
    )
    @test offsets_are_nonoverlapping(
        offsets;
        ignore_sentinel=true,
        ignore_empty=true,
    )

    @test first(result.ids) == token_to_id(tokenizer, "<s>")
    @test last(result.ids) == token_to_id(tokenizer, "</s>")
    @test first(offsets) == sentinel
    @test last(offsets) == sentinel
    @test first(masks) == 1
    @test last(masks) == 1

    empty_non_special = Tuple{Int,Int}[]
    for (offset, mask) in zip(offsets, masks)
        mask == 0 || continue
        offset == sentinel && continue
        if offset[1] == offset[2]
            push!(empty_non_special, offset)
        end
    end

    @test !isempty(empty_non_special)
    @test all(offset != sentinel for offset in empty_non_special)
    @test all(offset[1] >= offsets_index_base() for offset in empty_non_special)
end
