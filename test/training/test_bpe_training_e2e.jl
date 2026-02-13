@testset "BPE training end-to-end" begin
    corpus = [
        "hello world hello world",
        "tokenizers train subwords",
        "subwords train tokenizers",
        "offset checks stay stable",
        "stable deterministic merges",
    ]

    config_kwargs = (
        vocab_size=80,
        min_frequency=1,
        special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>"),
        end_of_word_marker="</w>",
        model_name="training_e2e_bpe",
    )

    training = train_bpe_result(corpus; config_kwargs...)
    tokenizer = training.tokenizer
    @test tokenizer isa BPETokenizer
    @test vocab_size(tokenizer) <= config_kwargs.vocab_size
    @test training.config.model_name == "training_e2e_bpe"
    @test training.artifacts.vocab_tokens == tokenizer.vocab.id_to_token

    samples = [
        "hello world",
        "tokenizers train subwords",
        "stable deterministic merges",
    ]

    for text in samples
        encoded = encode(tokenizer, text)
        @test decode(tokenizer, encoded) == text
        @test tokenize(tokenizer, text) == String[id_to_token(tokenizer, id) for id in encoded]
    end

    outdir = mktempdir()
    save_tokenizer(tokenizer, outdir)
    reloaded = load_tokenizer(outdir; format=:bpe)

    for text in samples
        @test tokenize(reloaded, text) == tokenize(tokenizer, text)
        @test encode(reloaded, text) == encode(tokenizer, text)
        @test decode(reloaded, encode(reloaded, text)) == decode(tokenizer, encode(tokenizer, text))
    end

    second = train_bpe_result(corpus; config_kwargs...)
    @test second.artifacts.vocab_tokens == training.artifacts.vocab_tokens
    @test second.artifacts.merge_pairs == training.artifacts.merge_pairs
end
