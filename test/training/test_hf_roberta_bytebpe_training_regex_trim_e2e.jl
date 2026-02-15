@testset "HF RoBERTa ByteBPE training regex+trim end-to-end" begin
    corpus = [
        "Hello my friend, how is your day going?",
        "Hello there\nHello there",
        "Hello there      dear",
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
        add_prefix_space=true,
        trim_offsets=true,
        use_regex=true,
        nfkc=false,
        lowercase=false,
        model_name="training_hf_roberta_bytebpe_regex_trim_e2e",
        version=v"0.3.0",
    )

    outdir = mktempdir()
    export_tokenizer(tokenizer, outdir; format=:hf_tokenizer_json)
    reloaded = load_hf_tokenizer_json(joinpath(outdir, "tokenizer.json"))
    @test reloaded isa HuggingFaceJSONTokenizer
    @test reloaded.base isa ByteBPETokenizer

    samples = [
        "Hello my friend, how is your day going?",
        "Hello there\nHello there",
        "Hello there      dear",
        "café 🙂",
    ]

    for text in samples
        @test tokenize(reloaded, text) == tokenize(tokenizer, text)
        for add_special_tokens in (false, true)
            ids_original = encode(tokenizer, text; add_special_tokens=add_special_tokens)
            ids_reloaded = encode(reloaded, text; add_special_tokens=add_special_tokens)
            @test ids_reloaded == ids_original
            @test decode(reloaded, ids_reloaded) == decode(tokenizer, ids_original)
        end
    end

    trim_text = "Hello there      dear"
    tokenization_text = tokenization_view(tokenizer, trim_text)
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

    saw_empty_non_special = false
    saw_trimmed_nonempty = false
    for (offset, mask) in zip(offsets, masks)
        mask == 0 || continue
        if offset == sentinel
            continue
        end

        if offset[1] == offset[2]
            saw_empty_non_special = true
            continue
        end

        span = try_span_substring(tokenization_text, offset)
        @test span !== nothing
        isempty(span) && continue
        @test !isspace(span[firstindex(span)])
        @test !isspace(span[lastindex(span)])
        saw_trimmed_nonempty = true
    end

    @test saw_empty_non_special
    @test saw_trimmed_nonempty
end
