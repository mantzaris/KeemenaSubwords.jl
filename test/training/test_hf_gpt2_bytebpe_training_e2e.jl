@testset "HF GPT-2 ByteBPE training end-to-end" begin
    corpus = [
        "Hello my friend, how is your day going?",
        "Hello there\nHello there",
        "Hello there      dear",
        "café costs 5 euros",
        "emoji 🙂 token",
        "<|endoftext|> appears in text",
    ]

    training = train_hf_gpt2_bytebpe_result(
        corpus;
        vocab_size=384,
        min_frequency=1,
        special_tokens=Dict(:unk => "<|endoftext|>"),
        end_of_word_marker="</w>",
        add_prefix_space=false,
        trim_offsets=true,
        use_regex=true,
        export_unk_token_null=true,
        model_name="training_hf_gpt2_bytebpe_e2e",
        version=v"0.3.0",
    )

    tokenizer = training.tokenizer
    @test tokenizer isa HuggingFaceJSONTokenizer
    @test tokenizer.base isa ByteBPETokenizer
    @test tokenizer.model isa KeemenaSubwords.HFBPEModelSpec
    @test tokenizer.model.unk_token === nothing
    @test tokenizer.normalizer isa KeemenaSubwords.HFNoopNormalizer
    @test tokenizer.pretokenizer isa KeemenaSubwords.HFByteLevelPreTokenizer
    @test tokenizer.postprocessor isa KeemenaSubwords.HFByteLevelPostProcessor
    @test tokenizer.decoder isa KeemenaSubwords.HFByteLevelDecoder
    @test training.artifacts.inner isa KeemenaSubwords.Training.ByteBPETrainingArtifacts
    @test !isempty(training.artifacts.hf_added_tokens)

    samples = [
        "Hello my friend, how is your day going?",
        "Hello there\nHello there",
        "Hello there      dear",
        "café 🙂",
    ]

    for text in samples
        ids_plain = encode(tokenizer, text; add_special_tokens=false)
        decoded_plain = decode(tokenizer, ids_plain)
        @test !isempty(decoded_plain)

        ids_with_specials = encode(tokenizer, text; add_special_tokens=true)
        @test ids_with_specials == ids_plain
        @test decode(tokenizer, ids_with_specials) == decoded_plain
    end

    tokens = tokenize(tokenizer, "Hello, world!")
    @test !isempty(tokens)
end
