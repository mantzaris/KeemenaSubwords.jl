@testset "HF BERT WordPiece training offsets contract" begin
    corpus = [
        "Hello, world!",
        "Café naïve façade",
        "你好 世界",
        "Hello world offsets",
    ]

    tokenizer = train_hf_bert_wordpiece(
        corpus;
        vocab_size=128,
        min_frequency=1,
        special_tokens=Dict(
            :unk => "[UNK]",
            :pad => "[PAD]",
            :cls => "[CLS]",
            :sep => "[SEP]",
            :mask => "[MASK]",
        ),
        continuation_prefix="##",
        max_input_chars_per_word=100,
        clean_text=true,
        handle_chinese_chars=true,
        lowercase=true,
        strip_accents=nothing,
        model_name="training_hf_bert_wordpiece_offsets",
    )

    samples = [
        "Hello,world!",
        "Café naïve façade",
        "你好 世界",
    ]

    sentinel = offsets_sentinel()
    cls_id = token_to_id(tokenizer, "[CLS]")
    sep_id = token_to_id(tokenizer, "[SEP]")
    continuation_prefix = tokenizer.base.continuation_prefix
    unk_token = id_to_token(tokenizer, unk_id(tokenizer))

    for clean_text in samples
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

        @test length(result.ids) == length(result.tokens) == length(offsets)
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

        @test first(result.ids) == cls_id
        @test last(result.ids) == sep_id
        @test first(offsets) == sentinel
        @test last(offsets) == sentinel

        for (token, offset) in zip(result.tokens, offsets)
            has_nonempty_span(offset) || continue
            span = try_span_substring(tokenization_text, offset)
            @test span !== nothing

            if token == unk_token
                @test !isempty(span)
            else
                expected = startswith(token, continuation_prefix) ?
                    replace(token, continuation_prefix => "") :
                    token
                @test span == expected
            end
        end
    end
end
