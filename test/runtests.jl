using Test
using KeemenaSubwords

const FIXTURES_DIR = joinpath(@__DIR__, "fixtures")
fixture(parts...) = joinpath(FIXTURES_DIR, parts...)

@testset "KeemenaSubwords sections 1-7" begin
    @testset "Model registry" begin
        names = available_models()
        @test :core_bpe_en in names
        @test :core_wordpiece_en in names
        @test :core_sentencepiece_unigram_en in names

        info = describe_model(:core_bpe_en)
        @test info.format == :bpe
        @test info.exists
        @test isdir(model_path(:core_bpe_en))
    end

    @testset "Classic BPE core model" begin
        tokenizer = load_tokenizer(:core_bpe_en)
        @test tokenizer isa BPETokenizer
        @test level_key(tokenizer) == Symbol(typeof(tokenizer))

        pieces = tokenize(tokenizer, "hello world")
        @test pieces == ["hello</w>", "world</w>"]

        ids = encode(tokenizer, "hello world"; add_special_tokens=true)
        @test first(ids) == bos_id(tokenizer)
        @test last(ids) == eos_id(tokenizer)
        @test decode(tokenizer, ids) == "hello world"

        vocab_path = joinpath(model_path(:core_bpe_en), "vocab.txt")
        merges_path = joinpath(model_path(:core_bpe_en), "merges.txt")

        t2 = load_tokenizer((vocab_path, merges_path))
        @test t2 isa BPETokenizer
        @test tokenize(t2, "keemena") == ["ke", "em", "ena</w>"]

        t3 = load_tokenizer((format=:bpe, vocab=vocab_path, merges=merges_path))
        @test t3 isa BPETokenizer
    end

    @testset "Byte-level BPE" begin
        tokenizer = load_tokenizer(model_path(:core_bpe_en); format=:bytebpe)
        @test tokenizer isa ByteBPETokenizer

        pieces = tokenize(tokenizer, "hello world")
        @test pieces == ["hello</w>", "world</w>"]

        ids = encode(tokenizer, "hello world")
        @test decode(tokenizer, ids) == "hello world"

        vocab_path = joinpath(model_path(:core_bpe_en), "vocab.txt")
        merges_path = joinpath(model_path(:core_bpe_en), "merges.txt")
        t2 = load_tokenizer((vocab_path, merges_path); format=:bytebpe)
        @test t2 isa ByteBPETokenizer
    end

    @testset "WordPiece core model" begin
        tokenizer = load_tokenizer(:core_wordpiece_en)

        @test tokenizer isa WordPieceTokenizer
        @test model_info(tokenizer).format == :wordpiece
        @test vocab_size(tokenizer) >= 10
        @test unk_id(tokenizer) == token_to_id(tokenizer, "[UNK]")

        @test tokenize(tokenizer, "hello keemena subwords") == [
            "hello", "ke", "##em", "##ena", "sub", "##word", "##s",
        ]

        encoded = encode(tokenizer, "hello keemena"; add_special_tokens=true)
        decoded = decode(tokenizer, encoded)
        @test decoded == "hello keemena"

        wp_dir = dirname(model_path(:core_wordpiece_en))
        t2 = load_tokenizer(wp_dir)
        @test t2 isa WordPieceTokenizer

        t3 = load_tokenizer((format=:wordpiece, path=model_path(:core_wordpiece_en)))
        @test t3 isa WordPieceTokenizer
    end

    @testset "Unigram direct loading" begin
        path = fixture("unigram", "unigram.tsv")
        tokenizer = load_tokenizer(path; format=:unigram)
        @test tokenizer isa UnigramTokenizer
        @test tokenize(tokenizer, "hello world") == ["hello", "world"]
        @test decode(tokenizer, encode(tokenizer, "hello world"; add_special_tokens=true)) == "helloworld"
    end

    @testset "SentencePiece Unigram core model" begin
        tokenizer = load_tokenizer(:core_sentencepiece_unigram_en)
        @test tokenizer isa SentencePieceTokenizer
        @test model_info(tokenizer).format == :sentencepiece

        pieces = tokenize(tokenizer, "hello world")
        @test pieces == ["▁hello", "▁world"]

        ids = encode(tokenizer, "hello world"; add_special_tokens=true)
        @test decode(tokenizer, ids) == "hello world"

        sp_path = model_path(:core_sentencepiece_unigram_en)
        t2 = load_tokenizer(sp_path)
        @test t2 isa SentencePieceTokenizer
    end

    @testset "SentencePiece BPE compatibility" begin
        sp_model = fixture("sentencepiece", "toy_bpe.model")
        tokenizer = load_tokenizer(sp_model)
        @test tokenizer isa SentencePieceTokenizer
        @test tokenize(tokenizer, "hello") == ["▁hello"]
        @test decode(tokenizer, encode(tokenizer, "hello"; add_special_tokens=true)) == "hello"
    end

    @testset "Section 5 deterministic fixture models" begin
        bpe = load_tokenizer(fixture("bpe"); format=:bpe)
        @test tokenize(bpe, "hello world") == ["hello</w>", "world</w>"]

        wp = load_tokenizer(fixture("wordpiece", "vocab.txt"); format=:wordpiece)
        @test tokenize(wp, "hello keemena subwords") == [
            "hello", "ke", "##em", "##ena", "sub", "##word", "##s",
        ]

        uni = load_tokenizer(fixture("unigram", "unigram.tsv"); format=:unigram)
        @test tokenize(uni, "hello world") == ["hello", "world"]
    end

    @testset "Section 6 IO format detection contracts" begin
        gpt2 = load_tokenizer(fixture("bpe_gpt2"); format=:auto)
        @test gpt2 isa ByteBPETokenizer
        @test tokenize(gpt2, "hello world") == ["hello", "world"]

        gpt2_explicit = load_tokenizer(fixture("bpe_gpt2"); format=:bpe_gpt2)
        @test gpt2_explicit isa ByteBPETokenizer

        @test_throws ArgumentError load_tokenizer(fixture("internal", "tokenizer.json"); format=:auto)
    end

    @testset "Section 4 integration helpers" begin
        tokenizer = load_tokenizer(:core_wordpiece_en)
        fn = keemena_callable(tokenizer)
        @test fn isa Function
        @test fn === tokenizer

        plain = (text::AbstractString) -> split(String(text))
        wrapped = keemena_callable(plain)
        @test wrapped === plain

        @test level_key(tokenizer) == Symbol(typeof(tokenizer))
        @test level_key(plain) == Symbol(typeof(plain))

        vocab_path = joinpath(model_path(:core_bpe_en), "vocab.txt")
        merges_path = joinpath(model_path(:core_bpe_en), "merges.txt")
        bpe = load_tokenizer((format=:bpe_gpt2, vocab=vocab_path, merges=merges_path))
        @test bpe isa BPETokenizer
    end

    @testset "Section 4 save/export APIs" begin
        wp = load_tokenizer(:core_wordpiece_en)
        bpe = load_tokenizer(:core_bpe_en)
        sp = load_tokenizer(:core_sentencepiece_unigram_en)

        tmp = mktempdir()

        wp_out = joinpath(tmp, "wp_internal")
        save_tokenizer(wp, wp_out)
        wp_reload = load_tokenizer(wp_out)
        @test tokenize(wp_reload, "hello keemena subwords") == tokenize(wp, "hello keemena subwords")

        bpe_out = joinpath(tmp, "bpe_export")
        export_tokenizer(bpe, bpe_out; format=:bpe_gpt2)
        bpe_reload = load_tokenizer((joinpath(bpe_out, "vocab.txt"), joinpath(bpe_out, "merges.txt")); format=:bpe_gpt2)
        @test tokenize(bpe_reload, "hello world") == tokenize(bpe, "hello world")

        sp_out = joinpath(tmp, "sp_export")
        export_tokenizer(sp, sp_out; format=:sentencepiece_model)
        sp_reload = load_tokenizer(joinpath(sp_out, "spm.model"))
        @test tokenize(sp_reload, "hello world") == tokenize(sp, "hello world")
    end

    @testset "Section 7 training implementations" begin
        corpus = ["hello world", "keemena subwords"]

        bpe = train_bpe(corpus; vocab_size=40, min_frequency=1)
        @test bpe isa BPETokenizer
        @test vocab_size(bpe) <= 40
        @test !isempty(tokenize(bpe, "hello world"))
        @test !isempty(encode(bpe, "hello world"))

        bpe_dir = mktempdir()
        save_tokenizer(bpe, bpe_dir)
        bpe_reload = load_tokenizer(bpe_dir; format=:bpe)
        @test !isempty(tokenize(bpe_reload, "hello world"))

        uni = train_unigram(corpus; vocab_size=40, seed_size=200, num_iters=2)
        @test uni isa UnigramTokenizer
        @test vocab_size(uni) <= 40
        @test !isempty(tokenize(uni, "hello world"))
        @test !isempty(encode(uni, "hello world"))

        uni_pruned = train_unigram(corpus; vocab_size=20, seed_size=400, num_iters=3)
        @test uni_pruned isa UnigramTokenizer
        @test vocab_size(uni_pruned) <= 20

        uni_dir = mktempdir()
        save_tokenizer(uni, uni_dir)
        uni_reload = load_tokenizer(uni_dir; format=:unigram)
        @test !isempty(tokenize(uni_reload, "hello world"))

        @test_throws ArgumentError train_wordpiece(corpus; vocab_size=100)
    end
end
