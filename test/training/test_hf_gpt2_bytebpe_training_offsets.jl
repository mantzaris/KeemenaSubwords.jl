@testset "HF GPT-2 ByteBPE training offsets contract" begin
    corpus = [
        "Hello my friend, how is your day going?",
        "Hello there\nHello there",
        "Hello there      dear",
        "café costs 5 euros",
        "emoji 🙂 token",
    ]

    tokenizer = train_hf_gpt2_bytebpe(
        corpus;
        vocab_size=384,
        min_frequency=1,
        special_tokens=Dict(:unk => "<|endoftext|>"),
        end_of_word_marker="</w>",
        add_prefix_space=false,
        trim_offsets=true,
        use_regex=true,
        export_unk_token_null=true,
        model_name="training_hf_gpt2_bytebpe_offsets",
        version=v"0.3.0",
    )

    samples = [
        "Hello my friend, how is your day going?",
        "Hello there\nHello there",
        "Hello there      dear",
        "café 🙂",
    ]

    for text in samples
        tokenization_text = tokenization_view(tokenizer, text)
        result = encode_result(
            tokenizer,
            tokenization_text;
            assume_normalized=true,
            add_special_tokens=true,
            return_offsets=true,
            return_masks=true,
        )

        @test result.offsets !== nothing
        @test result.special_tokens_mask !== nothing
        offsets = result.offsets

        @test length(result.ids) == length(result.tokens) == length(offsets) == length(result.special_tokens_mask)
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
    end
end
