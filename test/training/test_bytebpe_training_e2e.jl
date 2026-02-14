@testset "Byte-level BPE training end-to-end" begin
    corpus = [
        "hello, world!",
        "byte-level bpe test.",
        "café costs €5",
        "emoji 😀 party",
        "symbols [] {} <>",
    ]

    config_kwargs = (
        vocab_size=320,
        min_frequency=1,
        special_tokens=Dict(
            :unk => "<UNK>",
            :pad => "<PAD>",
            :bos => "<s>",
            :eos => "</s>",
        ),
        end_of_word_marker="</w>",
        include_full_byte_alphabet=true,
        model_name="training_e2e_bytebpe",
        version=v"0.3.0",
    )

    training = train_bytebpe_result(corpus; config_kwargs...)
    tokenizer = training.tokenizer
    @test tokenizer isa ByteBPETokenizer
    @test vocab_size(tokenizer) <= config_kwargs.vocab_size
    @test training.config.include_full_byte_alphabet
    @test training.artifacts.vocab_tokens == tokenizer.base.vocab.id_to_token
    @test training.artifacts.pair_ranks == tokenizer.base.pair_ranks
    for (rank, pair) in enumerate(training.artifacts.merge_pairs)
        @test training.artifacts.pair_ranks[pair] == rank
    end

    for byte in (0x00, 0x20, 0x7F, 0xC3, 0xFF)
        symbol = string(tokenizer.byte_to_unicode[byte + 1])
        @test haskey(tokenizer.base.vocab.token_to_id, symbol)
    end

    samples = [
        "hello, world!",
        "café costs €5",
        "emoji 😀 party",
    ]

    for text in samples
        encoded = encode(tokenizer, text; add_special_tokens=true)
        @test decode(tokenizer, encoded) == text
        @test tokenize(tokenizer, text) == String[id_to_token(tokenizer, id) for id in encode(tokenizer, text)]
    end

    outdir = mktempdir()
    save_tokenizer(tokenizer, outdir; format=:bpe)
    reloaded = load_tokenizer(
        outdir;
        format=:bytebpe,
        unk_token=training.config.special_tokens[:unk],
        end_of_word_marker=training.config.end_of_word_marker,
    )
    @test unk_id(reloaded) == unk_id(tokenizer)
    @test pad_id(tokenizer) !== nothing
    @test pad_id(reloaded) == pad_id(tokenizer)
    @test pad_id(reloaded) !== nothing

    for text in samples
        @test tokenize(reloaded, text) == tokenize(tokenizer, text)
        @test encode(reloaded, text) == encode(tokenizer, text)
        @test decode(reloaded, encode(reloaded, text; add_special_tokens=true)) == text
    end

    second = train_bytebpe_result(corpus; config_kwargs...)
    @test second.artifacts.vocab_tokens == training.artifacts.vocab_tokens
    @test second.artifacts.merge_pairs == training.artifacts.merge_pairs
    @test second.artifacts.pair_ranks == training.artifacts.pair_ranks
end
