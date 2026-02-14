@testset "WordPiece training end-to-end" begin
    corpus = [
        "hello hello world",
        "hello world hello",
        "wordpiece training keeps behavior stable",
        "tokenizer tests, tokenizer tests",
        "café costs five",
        "café cafe",
    ]

    config_kwargs = (
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
        model_name="training_e2e_wordpiece",
        version=v"0.3.0",
    )

    training = train_wordpiece_result(corpus; config_kwargs...)
    tokenizer = training.tokenizer
    @test tokenizer isa WordPieceTokenizer
    @test vocab_size(tokenizer) <= config_kwargs.vocab_size
    @test training.config.model_name == "training_e2e_wordpiece"
    @test training.artifacts.vocab_tokens == tokenizer.vocab.id_to_token
    @test length(training.artifacts.merge_pairs) == length(training.artifacts.merge_scores)
    @test !isempty(training.artifacts.word_counts)

    vocab_token_set = Set(training.artifacts.vocab_tokens)
    for required_token in ("h", "##h", "é", "##é", "[UNK]", "[CLS]", "[SEP]")
        @test required_token in vocab_token_set
    end

    samples = [
        "hello world",
        "hello   world",
        "tokenizer tests, café",
        "café costs five",
    ]

    for text in samples
        plain_ids = encode(tokenizer, text; add_special_tokens=false)
        @test tokenize(tokenizer, text) == String[id_to_token(tokenizer, id) for id in plain_ids]
        @test decode(tokenizer, plain_ids) == join(collect(eachsplit(text)), " ")
    end

    @test length(tokenize(tokenizer, "hello")) <= length(collect("hello"))

    outdir = mktempdir()
    save_tokenizer(tokenizer, outdir)
    reloaded = load_tokenizer(outdir; format=:wordpiece)

    for text in samples
        @test tokenize(reloaded, text) == tokenize(tokenizer, text)
        @test encode(reloaded, text; add_special_tokens=true) == encode(tokenizer, text; add_special_tokens=true)
        @test decode(reloaded, encode(reloaded, text; add_special_tokens=true)) ==
              decode(tokenizer, encode(tokenizer, text; add_special_tokens=true))
    end

    export_dir = mktempdir()
    export_tokenizer(tokenizer, export_dir; format=:wordpiece_vocab)
    reloaded_export = load_wordpiece(export_dir)
    for text in samples
        @test tokenize(reloaded_export, text) == tokenize(tokenizer, text)
        @test encode(reloaded_export, text; add_special_tokens=false) == encode(tokenizer, text; add_special_tokens=false)
    end

    second = train_wordpiece_result(corpus; config_kwargs...)
    @test second.artifacts.vocab_tokens == training.artifacts.vocab_tokens
    @test second.artifacts.merge_pairs == training.artifacts.merge_pairs
    @test second.artifacts.merge_scores == training.artifacts.merge_scores

    reversed_training = train_wordpiece_result(reverse(corpus); config_kwargs...)
    @test reversed_training.artifacts.vocab_tokens == training.artifacts.vocab_tokens
    @test reversed_training.artifacts.merge_pairs == training.artifacts.merge_pairs
    @test reversed_training.artifacts.merge_scores == training.artifacts.merge_scores
end
