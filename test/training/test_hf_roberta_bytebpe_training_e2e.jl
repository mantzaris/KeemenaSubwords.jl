@testset "HF RoBERTa ByteBPE training end-to-end" begin
    corpus = [
        "hello world",
        "hello, world!",
        "byte level   bpe",
        "café costs 5 euros",
        "emoji 🙂 token",
        "hello world hello world",
    ]

    training = train_hf_roberta_bytebpe_result(
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
        model_name="training_hf_roberta_bytebpe_e2e",
        version=v"0.3.0",
    )

    tokenizer = training.tokenizer
    @test tokenizer isa HuggingFaceJSONTokenizer
    @test tokenizer.base isa ByteBPETokenizer
    @test tokenizer.model isa KeemenaSubwords.HFBPEModelSpec
    @test tokenizer.normalizer isa KeemenaSubwords.HFNoopNormalizer
    @test tokenizer.pretokenizer isa KeemenaSubwords.HFByteLevelPreTokenizer
    @test tokenizer.postprocessor isa KeemenaSubwords.HFRobertaProcessingPostProcessor
    @test tokenizer.decoder isa KeemenaSubwords.HFByteLevelDecoder
    @test training.artifacts.inner isa KeemenaSubwords.Training.ByteBPETrainingArtifacts
    @test !isempty(training.artifacts.hf_added_tokens)
    @test all(token.special for token in training.artifacts.hf_added_tokens)
    @test all(!token.normalized for token in training.artifacts.hf_added_tokens)

    bos = token_to_id(tokenizer, "<s>")
    eos = token_to_id(tokenizer, "</s>")

    roundtrip_samples = [
        "hello world",
        "café costs",
        "emoji 🙂 token",
    ]

    for text in roundtrip_samples
        ids_plain = encode(tokenizer, text; add_special_tokens=false)
        @test decode(tokenizer, ids_plain) == text

        ids_special = encode(tokenizer, text; add_special_tokens=true)
        @test first(ids_special) == bos
        @test last(ids_special) == eos
        @test decode(tokenizer, ids_special) == text
    end

    punctuation_tokens = tokenize(tokenizer, "hello, world!")
    @test !isempty(punctuation_tokens)
end
