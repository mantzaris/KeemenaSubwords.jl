@testset "HF RoBERTa ByteBPE training offsets contract" begin
    corpus = [
        "hello world",
        "hello, world!",
        "byte level bpe",
        "café costs 5 euros",
        "emoji 🙂 token",
    ]

    tokenizer = train_hf_roberta_bytebpe(
        corpus;
        vocab_size=384,
        min_frequency=1,
        special_tokens=Dict(
            :unk => "<unk>",
            :pad => "<pad>",
            :bos => "<s>",
            :eos => "</s>",
            :mask => "<mask>",
        ),
        end_of_word_marker="</w>",
        add_prefix_space=false,
        trim_offsets=false,
        use_regex=false,
        nfkc=false,
        lowercase=false,
        model_name="training_hf_roberta_bytebpe_offsets",
        version=v"0.3.0",
    )

    samples = [
        "hello world",
        "café costs 5",
        "emoji 🙂 token",
    ]

    bos = token_to_id(tokenizer, "<s>")
    eos = token_to_id(tokenizer, "</s>")
    sentinel = offsets_sentinel()

    for text in samples
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

        @test length(result.ids) == length(result.tokens) == length(offsets) == length(masks)
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

        @test first(result.ids) == bos
        @test last(result.ids) == eos
        @test first(offsets) == sentinel
        @test last(offsets) == sentinel
        @test first(masks) == 1
        @test last(masks) == 1
    end
end
