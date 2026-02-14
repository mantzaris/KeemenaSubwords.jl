@testset "BPE training end-to-end" begin
    corpus = [
        "hello, world!",
        "tokenizers train subwords quickly",
        "prices [usd] are 12.50",
        "café costs €5",
        "emoji 😀 party",
        "brackets (x) [y] {z}",
    ]

    config_kwargs = (
        vocab_size=128,
        min_frequency=1,
        special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>"),
        end_of_word_marker="</w>",
        model_name="training_e2e_bpe",
        version=v"0.3.0",
    )

    training = train_bpe_result(corpus; config_kwargs...)
    tokenizer = training.tokenizer
    @test tokenizer isa BPETokenizer
    @test vocab_size(tokenizer) <= config_kwargs.vocab_size
    @test training.config.model_name == "training_e2e_bpe"
    @test training.artifacts.vocab_tokens == tokenizer.vocab.id_to_token
    @test training.artifacts.pair_ranks == tokenizer.pair_ranks

    vocab_token_set = Set(training.artifacts.vocab_tokens)
    for (rank, pair) in enumerate(training.artifacts.merge_pairs)
        @test training.artifacts.pair_ranks[pair] == rank
        @test pair[1] * pair[2] in vocab_token_set
    end

    samples = [
        "hello, world!",
        "prices [usd] are 12.50",
        "café costs €5",
        "emoji 😀 party",
    ]

    for text in samples
        encoded = encode(tokenizer, text)
        @test decode(tokenizer, encoded) == text
        @test tokenize(tokenizer, text) == String[id_to_token(tokenizer, id) for id in encoded]
    end

    outdir = mktempdir()
    save_tokenizer(tokenizer, outdir)
    reloaded = load_tokenizer(outdir; format=:bpe)
    @test unk_id(reloaded) == unk_id(tokenizer)
    @test pad_id(tokenizer) !== nothing
    @test pad_id(reloaded) == pad_id(tokenizer)
    @test pad_id(reloaded) !== nothing

    for text in samples
        @test tokenize(reloaded, text) == tokenize(tokenizer, text)
        @test encode(reloaded, text) == encode(tokenizer, text)
        @test decode(reloaded, encode(reloaded, text)) == decode(tokenizer, encode(tokenizer, text))
    end

    second = train_bpe_result(corpus; config_kwargs...)
    @test second.artifacts.vocab_tokens == training.artifacts.vocab_tokens
    @test second.artifacts.merge_pairs == training.artifacts.merge_pairs
    @test second.artifacts.pair_ranks == training.artifacts.pair_ranks

    reversed_training = train_bpe_result(reverse(corpus); config_kwargs...)
    @test reversed_training.artifacts.vocab_tokens == training.artifacts.vocab_tokens
    @test reversed_training.artifacts.merge_pairs == training.artifacts.merge_pairs
    @test reversed_training.artifacts.pair_ranks == training.artifacts.pair_ranks
end
