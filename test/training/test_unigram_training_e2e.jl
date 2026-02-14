@testset "Unigram training end-to-end" begin
    corpus = [
        "hello, world!",
        "tokenizers [train] unigram models",
        "prices are 12.50 usd",
        "café costs €5",
        "emoji 😀 party",
        "brackets (x) [y] {z}",
    ]

    config_kwargs = (
        vocab_size=128,
        seed_size=1200,
        num_iters=3,
        max_subword_length=8,
        prune_fraction=0.2,
        special_tokens=Dict(
            :unk => "<UNK>",
            :pad => "<PAD>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        whitespace_marker="▁",
        model_name="training_e2e_unigram",
        version=v"0.3.0",
    )

    training = train_unigram_result(corpus; config_kwargs...)
    tokenizer = training.tokenizer
    @test tokenizer isa UnigramTokenizer
    @test vocab_size(tokenizer) <= config_kwargs.vocab_size
    @test training.artifacts.vocab_tokens == tokenizer.vocab.id_to_token
    @test training.artifacts.token_logprobs == tokenizer.logprobs
    @test !isempty(training.artifacts.word_counts)

    samples = [
        "hello, world!",
        "prices are 12.50 usd",
        "café costs €5",
        "emoji 😀 party",
    ]

    for text in samples
        plain_ids = encode(tokenizer, text; add_special_tokens=false)
        @test tokenize(tokenizer, text) == String[id_to_token(tokenizer, id) for id in plain_ids]

        ids = encode(tokenizer, text; add_special_tokens=true)
        @test decode(tokenizer, ids) == text
    end

    outdir = mktempdir()
    save_tokenizer(tokenizer, outdir)
    reloaded = load_tokenizer(outdir; format=:unigram)

    for text in samples
        @test tokenize(reloaded, text) == tokenize(tokenizer, text)
        @test encode(reloaded, text; add_special_tokens=true) == encode(tokenizer, text; add_special_tokens=true)
        @test decode(reloaded, encode(reloaded, text; add_special_tokens=true)) == decode(tokenizer, encode(tokenizer, text; add_special_tokens=true))
    end

    second = train_unigram_result(corpus; config_kwargs...)
    @test second.artifacts.vocab_tokens == training.artifacts.vocab_tokens
    @test all(isapprox.(second.artifacts.token_logprobs, training.artifacts.token_logprobs; atol=1e-12, rtol=0.0))

    reversed_training = train_unigram_result(reverse(corpus); config_kwargs...)
    @test reversed_training.artifacts.vocab_tokens == training.artifacts.vocab_tokens
    @test all(isapprox.(reversed_training.artifacts.token_logprobs, training.artifacts.token_logprobs; atol=1e-12, rtol=0.0))
end

@testset "Unigram whitespace marker save/reload parity" begin
    corpus = [
        "hello world",
        "unigram marker test",
        "offset safe tokens",
    ]

    training = train_unigram_result(
        corpus;
        vocab_size=96,
        seed_size=800,
        num_iters=2,
        max_subword_length=6,
        prune_fraction=0.2,
        special_tokens=Dict(
            :unk => "<UNK>",
            :pad => "<PAD>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        whitespace_marker="▁",
        model_name="training_unigram_marker",
    )
    tokenizer = training.tokenizer
    @test tokenizer.whitespace_marker == "▁"

    sample = "hello world"
    original_tokens = tokenize(tokenizer, sample)
    original_ids = encode(tokenizer, sample; add_special_tokens=true)
    @test decode(tokenizer, original_ids) == sample

    outdir = mktempdir()
    save_tokenizer(tokenizer, outdir)
    reloaded = load_tokenizer(outdir; format=:unigram)
    @test reloaded isa UnigramTokenizer
    @test reloaded.whitespace_marker == "▁"
    @test tokenize(reloaded, sample) == original_tokens
    @test encode(reloaded, sample; add_special_tokens=true) == original_ids
    @test decode(reloaded, original_ids) == sample
end

@testset "Unigram sentencepiece export/load parity" begin
    corpus = [
        "hello, world!",
        "café costs €5",
        "markers keep spacing",
    ]

    training = train_unigram_result(
        corpus;
        vocab_size=96,
        seed_size=800,
        num_iters=2,
        max_subword_length=8,
        prune_fraction=0.2,
        special_tokens=Dict(
            :unk => "<UNK>",
            :pad => "<PAD>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        whitespace_marker="▁",
        model_name="training_unigram_spm_export",
    )
    tokenizer = training.tokenizer

    outdir = mktempdir()
    export_tokenizer(tokenizer, outdir; format=:sentencepiece_model)
    spm_path = joinpath(outdir, "spm.model")
    @test isfile(spm_path)

    reloaded = load_tokenizer(spm_path; format=:sentencepiece_model)
    @test reloaded isa SentencePieceTokenizer

    samples = [
        "hello, world!",
        "café costs €5",
    ]
    for text in samples
        @test tokenize(reloaded, text) == tokenize(tokenizer, text)
        @test encode(reloaded, text; add_special_tokens=false) == encode(tokenizer, text; add_special_tokens=false)
        @test decode(reloaded, encode(reloaded, text; add_special_tokens=true)) ==
              decode(tokenizer, encode(tokenizer, text; add_special_tokens=true))
    end
end
