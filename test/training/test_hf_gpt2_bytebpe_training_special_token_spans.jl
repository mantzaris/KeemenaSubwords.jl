@testset "HF GPT-2 ByteBPE training special token spans" begin
    corpus = [
        "Hello my friend, how is your day going?",
        "special token literal <|endoftext|>",
        "café costs 5 euros",
    ]

    tokenizer = train_hf_gpt2_bytebpe(
        corpus;
        vocab_size=320,
        min_frequency=1,
        special_tokens=Dict(:unk => "<|endoftext|>"),
        end_of_word_marker="</w>",
        add_prefix_space=false,
        trim_offsets=true,
        use_regex=true,
        export_unk_token_null=true,
        model_name="training_hf_gpt2_bytebpe_special_spans",
        version=v"0.3.0",
    )

    text = "start <|endoftext|> end"
    tokenization_text = tokenization_view(tokenizer, text)
    result = encode_result(
        tokenizer,
        tokenization_text;
        assume_normalized=true,
        add_special_tokens=false,
        return_offsets=true,
        return_masks=true,
    )

    @test result.offsets !== nothing
    @test result.special_tokens_mask !== nothing

    special_id = token_to_id(tokenizer, "<|endoftext|>")
    sentinel = offsets_sentinel()
    matched_positions = Int[]

    for (index, token_id) in enumerate(result.ids)
        token_id == special_id || continue
        push!(matched_positions, index)
        @test result.special_tokens_mask[index] == 1

        offset = result.offsets[index]
        @test offset != sentinel
        @test has_nonempty_span(offset)

        span = try_span_substring(tokenization_text, offset)
        @test span == "<|endoftext|>"
    end

    @test !isempty(matched_positions)
end
