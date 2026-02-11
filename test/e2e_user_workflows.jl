@testset "E2E user workflows" begin
    @testset "Workflow 1: HF tokenizer.json directory autodetect" begin
        hf_dir = fixture("hf_json_wordpiece")
        tok = load_tokenizer(hf_dir)
        @test tok isa HuggingFaceJSONTokenizer
        @test !isempty(tokenize(tok, "Hello world"))

        ids = encode(tok, "Hello world"; add_special_tokens=true)
        @test !isempty(ids)
        @test decode(tok, ids) == "hello world"

        result = encode_result(tok, "Hello world"; return_offsets=true, return_masks=true)
        @test result.ids == ids
        @test result.attention_mask !== nothing
        @test result.special_tokens_mask !== nothing
    end

    @testset "Workflow 2: register local model and load by key" begin
        local_key = Symbol("e2e_local_hf_" * string(time_ns()))
        register_local_model!(
            local_key,
            fixture("hf_json_wordpiece");
            format=:hf_tokenizer_json,
            family=:local,
            description="E2E local HF model",
        )

        tok = load_tokenizer(local_key; prefetch=false)
        @test tok isa HuggingFaceJSONTokenizer
        ids = encode(local_key, "Hello world"; prefetch=false)
        @test !isempty(ids)
        @test decode(local_key, ids; prefetch=false) == "hello world"
    end

    @testset "Workflow 3: WordPiece from vocab.txt fixture" begin
        vocab = fixture("wordpiece", "vocab.txt")
        tok = load_tokenizer(vocab; format=:wordpiece_vocab)
        @test tok isa WordPieceTokenizer
        ids = encode(vocab, "hello keemena"; format=:wordpiece_vocab)
        @test !isempty(ids)
        @test decode(vocab, ids; format=:wordpiece_vocab) == "hello keemena"
    end

    @testset "Workflow 4: SentencePiece discovery filename and single-model heuristic" begin
        named_dir = mktempdir()
        cp(fixture("sentencepiece", "toy_bpe.model"), joinpath(named_dir, "spiece.model"))
        @test detect_tokenizer_format(named_dir) == :sentencepiece_model
        @test load_tokenizer(named_dir) isa SentencePieceTokenizer

        heuristic_dir = mktempdir()
        cp(fixture("sentencepiece", "toy_bpe.model"), joinpath(heuristic_dir, "custom.model"))
        @test detect_tokenizer_format(heuristic_dir) == :sentencepiece_model
        @test load_tokenizer(heuristic_dir) isa SentencePieceTokenizer
    end

    if _DOWNLOAD_TESTS
        @testset "Workflow 5: built-in download and one-shot usage" begin
            keys = [
                :tiktoken_o200k_base,
                :openai_gpt2_bpe,
                :bert_base_uncased_wordpiece,
                :t5_small_sentencepiece_unigram,
                :qwen2_5_bpe,
            ]
            status = prefetch_models_status(keys)

            for key in keys
                st = status[key]
                @test st.available
                @test st.method in (:artifact, :fallback_download, :already_present)
                @test st.path !== nothing

                tok = load_tokenizer(key)
                ids = encode(tok, "Hello world")
                @test !isempty(ids)

                one_shot = if tok isa TiktokenTokenizer
                    encode_result(key, "Hello world"; add_special_tokens=false, return_masks=true)
                else
                    encode_result(key, "Hello world"; return_masks=true)
                end
                @test !isempty(one_shot.ids)
                @test one_shot.attention_mask !== nothing
                @test decode(tok, ids) isa String
            end
        end
    end
end
