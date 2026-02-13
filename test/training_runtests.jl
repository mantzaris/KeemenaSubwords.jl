@testset "Section 7 training implementations" begin
    corpus = ["hello world", "keemena subwords"]

    bpe = train_bpe(corpus; vocab_size=40, min_frequency=1)
    @test bpe isa BPETokenizer
    @test vocab_size(bpe) <= 40
    @test !isempty(tokenize(bpe, "hello world"))
    @test !isempty(encode(bpe, "hello world"))
    @test !isempty(decode(bpe, encode(bpe, "hello world")))

    bpe_dir = mktempdir()
    save_tokenizer(bpe, bpe_dir)
    bpe_reload = load_tokenizer(bpe_dir; format=:bpe)
    @test !isempty(tokenize(bpe_reload, "hello world"))

    uni = train_unigram(corpus; vocab_size=40, seed_size=200, num_iters=2)
    @test uni isa UnigramTokenizer
    @test vocab_size(uni) <= 40
    @test !isempty(tokenize(uni, "hello world"))
    @test !isempty(encode(uni, "hello world"))
    @test !isempty(decode(uni, encode(uni, "hello world")))

    uni_pruned = train_unigram(corpus; vocab_size=20, seed_size=400, num_iters=3)
    @test uni_pruned isa UnigramTokenizer
    @test vocab_size(uni_pruned) <= 20

    uni_dir = mktempdir()
    save_tokenizer(uni, uni_dir)
    uni_reload = load_tokenizer(uni_dir; format=:unigram)
    @test !isempty(tokenize(uni_reload, "hello world"))

    @test_throws ArgumentError train_wordpiece(corpus; vocab_size=100)
    @test_throws ArgumentError train_sentencepiece(corpus; vocab_size=100)

    training_heavy = get(ENV, "KEEMENA_TEST_TRAINING_HEAVY", "0") == "1"
    if training_heavy
        @testset "Section 7 training heavy (reserved)" begin
            @test true
        end
    end
end
