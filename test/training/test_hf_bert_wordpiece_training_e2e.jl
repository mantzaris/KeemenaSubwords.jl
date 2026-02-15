@testset "HF BERT WordPiece training end-to-end" begin
    corpus = [
        "Hello, world!",
        "Café naïve façade",
        "你好 世界",
        "Hello, world!",
        "Cafe naive facade",
        "你好 世界",
    ]

    training = train_hf_bert_wordpiece_result(
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
        model_name="training_hf_bert_wordpiece_e2e",
        version=v"0.3.0",
    )

    tokenizer = training.tokenizer
    @test tokenizer isa HuggingFaceJSONTokenizer
    @test tokenizer.base isa WordPieceTokenizer
    @test tokenizer.normalizer isa KeemenaSubwords.HFBertNormalizer
    @test tokenizer.pretokenizer isa KeemenaSubwords.HFBertPreTokenizer
    @test training.artifacts.inner isa KeemenaSubwords.Training.WordPieceTrainingArtifacts
    @test !isempty(training.artifacts.hf_added_tokens)
    @test all(token.special for token in training.artifacts.hf_added_tokens)
    @test all(!token.normalized for token in training.artifacts.hf_added_tokens)

    @test normalize(tokenizer, "Café") == "cafe"
    @test tokenize(tokenizer, "Hello,world!") == ["hello", ",", "world", "!"]

    cls_id = token_to_id(tokenizer, "[CLS]")
    sep_id = token_to_id(tokenizer, "[SEP]")

    special_in_text_ids = encode(tokenizer, "[CLS] hello"; add_special_tokens=false)
    @test !isempty(special_in_text_ids)
    @test first(special_in_text_ids) == cls_id

    processed_ids = encode(tokenizer, "hello world"; add_special_tokens=true)
    @test first(processed_ids) == cls_id
    @test last(processed_ids) == sep_id

    outdir = mktempdir()
    export_tokenizer(tokenizer, outdir; format=:hf_tokenizer_json)
    reloaded = load_hf_tokenizer_json(joinpath(outdir, "tokenizer.json"))
    @test reloaded isa HuggingFaceJSONTokenizer

    samples = [
        "Hello, world!",
        "Café naïve façade",
        "你好 世界",
        "[CLS] hello",
    ]

    for text in samples
        @test tokenize(reloaded, text) == tokenize(tokenizer, text)
        for add_special_tokens in (false, true)
            @test encode(reloaded, text; add_special_tokens=add_special_tokens) ==
                  encode(tokenizer, text; add_special_tokens=add_special_tokens)
        end

        ids_reloaded = encode(reloaded, text; add_special_tokens=true)
        ids_original = encode(tokenizer, text; add_special_tokens=true)
        @test decode(reloaded, ids_reloaded) == decode(tokenizer, ids_original)
    end
end
