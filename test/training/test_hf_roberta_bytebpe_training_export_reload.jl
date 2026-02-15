@testset "HF RoBERTa ByteBPE training export/reload parity" begin
    corpus = [
        "hello world",
        "hello, world!",
        "byte level bpe",
        "café costs 5 euros",
        "emoji 🙂 token",
        "deterministic merge ordering",
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
        model_name="training_hf_roberta_bytebpe_export_reload",
        version=v"0.3.0",
    )

    outdir = mktempdir()
    export_tokenizer(tokenizer, outdir; format=:hf_tokenizer_json)

    reloaded = load_hf_tokenizer_json(joinpath(outdir, "tokenizer.json"))
    @test reloaded isa HuggingFaceJSONTokenizer
    @test reloaded.base isa ByteBPETokenizer

    reloaded_auto = load_tokenizer(outdir; format=:auto)
    @test reloaded_auto isa HuggingFaceJSONTokenizer
    @test reloaded_auto.base isa ByteBPETokenizer

    samples = [
        "hello world",
        "hello, world!",
        "café costs 5",
        "emoji 🙂 token",
    ]

    for candidate in (reloaded, reloaded_auto)
        for text in samples
            @test tokenize(candidate, text) == tokenize(tokenizer, text)
            for add_special_tokens in (false, true)
                ids_original = encode(tokenizer, text; add_special_tokens=add_special_tokens)
                ids_reloaded = encode(candidate, text; add_special_tokens=add_special_tokens)
                @test ids_reloaded == ids_original
                @test decode(candidate, ids_reloaded) == decode(tokenizer, ids_original)
            end
        end
    end
end
