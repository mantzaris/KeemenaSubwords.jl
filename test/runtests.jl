using Test
using KeemenaSubwords

const FIXTURES_DIR = joinpath(@__DIR__, "fixtures")
fixture(parts...) = joinpath(FIXTURES_DIR, parts...)

@testset "KeemenaSubwords sections 1-15" begin
    @testset "Model registry" begin
        names = available_models()
        @test :core_bpe_en in names
        @test :core_wordpiece_en in names
        @test :core_sentencepiece_unigram_en in names
        @test :tiktoken_o200k_base in names
        @test :tiktoken_cl100k_base in names
        @test :openai_gpt2_bpe in names
        @test :bert_base_uncased_wordpiece in names
        @test :bert_base_multilingual_cased_wordpiece in names
        @test :t5_small_sentencepiece_unigram in names
        @test :mistral_v1_sentencepiece in names
        @test :mistral_v3_sentencepiece in names
        @test :phi2_bpe in names
        @test :qwen2_5_bpe in names
        @test :roberta_base_bpe in names
        @test :xlm_roberta_base_sentencepiece_bpe in names
        @test :llama2_tokenizer in names
        @test :llama3_8b_tokenizer in names

        info = describe_model(:core_bpe_en)
        @test info.format == :bpe
        @test info.exists
        @test isdir(model_path(:core_bpe_en))

        gpt2_info = describe_model(:openai_gpt2_bpe)
        @test gpt2_info.format == :bpe_gpt2
        @test length(gpt2_info.files) == 2
        @test all(isabspath, gpt2_info.files)

        mistral_info = describe_model(:mistral_v1_sentencepiece)
        @test mistral_info.license == "Apache-2.0"
        @test mistral_info.format == :sentencepiece_model

        bpe_models = available_models(format=:bpe_gpt2)
        @test :phi2_bpe in bpe_models
        @test :qwen2_5_bpe in bpe_models
        @test :roberta_base_bpe in bpe_models

        @test available_models(family=:qwen) == [:qwen2_5_bpe]
        @test :core_bpe_en in available_models(shipped=true)
        @test :qwen2_5_bpe in available_models(format=:hf_tokenizer_json)
        gated = available_models(distribution=:installable_gated)
        @test :llama2_tokenizer in gated
        @test :llama3_8b_tokenizer in gated
        llama_info = describe_model(:llama3_8b_tokenizer)
        @test llama_info.distribution == :installable_gated
        @test !llama_info.shipped
        @test :qwen2_5_bpe in recommended_defaults_for_llms()
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

    @testset "Section 11 Hugging Face tokenizer.json loader" begin
        hf = load_tokenizer(fixture("hf_json_wordpiece"); format=:auto)
        @test hf isa HuggingFaceJSONTokenizer
        @test tokenize(hf, "Hello keemena subwords") == [
            "hello", "ke", "##em", "##ena", "sub", "##word", "##s",
        ]

        ids = encode(hf, "Hello world"; add_special_tokens=true)
        @test first(ids) == token_to_id(hf, "[CLS]")
        @test last(ids) == token_to_id(hf, "[SEP]")
        @test decode(hf, ids) == "hello world"

        err = try
            load_tokenizer(fixture("hf_json_unsupported"); format=:hf_tokenizer_json)
            nothing
        catch ex
            ex
        end
        @test err isa ArgumentError
        @test occursin("Unsupported pre_tokenizer type", sprint(showerror, err))
        @test occursin("\$.pre_tokenizer", sprint(showerror, err))
        @test occursin("Workaround", sprint(showerror, err))
    end

    @testset "Section 15 Hugging Face compliance expansion" begin
        hf_bpe = load_hf_tokenizer_json(fixture("hf_json_bytelevel_bpe", "tokenizer.json"))
        @test hf_bpe isa HuggingFaceJSONTokenizer
        @test tokenize(hf_bpe, "Hello world") == ["hello</w>", "world</w>"]
        bpe_ids = encode(hf_bpe, "Hello world")
        @test !isempty(bpe_ids)
        @test decode(hf_bpe, bpe_ids) == "hello world"

        hf_uni = load_hf_tokenizer_json(fixture("hf_json_unigram_metaspace", "tokenizer.json"))
        @test hf_uni isa HuggingFaceJSONTokenizer
        uni_ids = encode(hf_uni, "Hello world")
        @test !isempty(uni_ids)
        @test decode(hf_uni, uni_ids) == "hello world"

        hf_added = load_hf_tokenizer_json(fixture("hf_json_added_tokens", "tokenizer.json"))
        @test hf_added isa HuggingFaceJSONTokenizer

        # normalized=true + lstrip/rstrip should match <CITY> around extra spaces
        ids_city = encode(hf_added, "I love   <city>   !")
        @test token_to_id(hf_added, "<CITY>") in ids_city

        # special added tokens must bypass normalization and match verbatim
        ids_special = encode(hf_added, "[SPECIAL] i")
        @test token_to_id(hf_added, "[SPECIAL]") == first(ids_special)

        # TemplateProcessing should prepend [SPECIAL] when requested
        templated = encode(hf_added, "i love !"; add_special_tokens=true)
        @test token_to_id(hf_added, "[SPECIAL]") == first(templated)
    end

    @testset "Section 9 built-in public baseline keys" begin
        availability = prefetch_models([
            :tiktoken_o200k_base,
            :tiktoken_cl100k_base,
            :openai_gpt2_bpe,
            :bert_base_uncased_wordpiece,
            :t5_small_sentencepiece_unigram,
        ])
        @test all(values(availability))

        for key in (
            :tiktoken_o200k_base,
            :tiktoken_cl100k_base,
            :tiktoken_r50k_base,
            :tiktoken_p50k_base,
        )
            tt = load_tokenizer(key)
            @test tt isa TiktokenTokenizer
            ids = encode(tt, "hello world")
            @test !isempty(ids)
            @test decode(tt, ids) == "hello world"
            @test !isempty(tokenize(tt, "hello world"))
        end

        gpt2 = load_tokenizer(:openai_gpt2_bpe)
        @test gpt2 isa ByteBPETokenizer
        gpt2_roundtrip = decode(gpt2, encode(gpt2, "hello world"))
        @test replace(gpt2_roundtrip, " " => "") == "helloworld"

        bert = load_tokenizer(:bert_base_uncased_wordpiece)
        @test bert isa WordPieceTokenizer
        @test !isempty(tokenize(bert, "hello keemena"))

        info = describe_model(:bert_base_multilingual_cased_wordpiece)
        @test info.distribution == :artifact_public
        @test "vocab.txt" in info.expected_files

        multi = prefetch_models([:bert_base_multilingual_cased_wordpiece])
        @test haskey(multi, :bert_base_multilingual_cased_wordpiece)
        if multi[:bert_base_multilingual_cased_wordpiece]
            bert_multi = load_tokenizer(:bert_base_multilingual_cased_wordpiece)
            @test bert_multi isa WordPieceTokenizer
            @test !isempty(tokenize(bert_multi, "hello mundo"))
        elseif get(ENV, "KEEMENA_RUN_NETWORK_TESTS", "0") == "1"
            # In explicit network mode this should be downloadable.
            @test multi[:bert_base_multilingual_cased_wordpiece]
        else
            @test info.format == :wordpiece_vocab
        end

        t5 = load_tokenizer(:t5_small_sentencepiece_unigram)
        @test t5 isa SentencePieceTokenizer
        @test !isempty(tokenize(t5, "hello world"))
    end

    @testset "Section 9 curated model inventory keys" begin
        pre = prefetch_models([:mistral_v1_sentencepiece])
        @test pre[:mistral_v1_sentencepiece]
        @test isdir(model_path(:mistral_v1_sentencepiece))

        curated = prefetch_models([
            :mistral_v1_sentencepiece,
            :mistral_v3_sentencepiece,
            :phi2_bpe,
            :qwen2_5_bpe,
            :roberta_base_bpe,
            :xlm_roberta_base_sentencepiece_bpe,
        ])
        @test all(values(curated))

        for key in (:mistral_v1_sentencepiece, :mistral_v3_sentencepiece, :xlm_roberta_base_sentencepiece_bpe)
            tok = load_tokenizer(key)
            @test tok isa SentencePieceTokenizer
            @test !isempty(tokenize(tok, "hello world"))
            ids = encode(tok, "hello world")
            @test !isempty(ids)
            @test decode(tok, ids) isa String
        end

        for key in (:phi2_bpe, :roberta_base_bpe)
            tok = load_tokenizer(key)
            @test tok isa ByteBPETokenizer
            pieces = tokenize(tok, "hello world")
            @test !isempty(pieces)
            ids = encode(tok, "hello world")
            @test !isempty(ids)
            @test decode(tok, ids) isa String
        end

        qwen = load_tokenizer(:qwen2_5_bpe)
        @test qwen isa HuggingFaceJSONTokenizer || qwen isa ByteBPETokenizer
        @test !isempty(tokenize(qwen, "hello world"))
        @test decode(qwen, encode(qwen, "hello world")) isa String
    end

    @testset "Section 10 external model registration" begin
        ext_key = :external_test_bpe
        ext_path = fixture("bpe")
        register_local_model!(
            ext_key,
            ext_path;
            format=:bpe,
            family=:external,
            description="External BPE fixture for testing",
        )

        @test ext_key in available_models(family=:external)
        tok = load_tokenizer(ext_key; prefetch=false)
        @test tok isa BPETokenizer
        @test !isempty(tokenize(tok, "hello world"))
    end

    @testset "Section 11 local registry and HF download helper" begin
        local_key = :local_test_wordpiece
        register_local_model!(
            local_key,
            fixture("wordpiece");
            format=:wordpiece_vocab,
            description="Local WordPiece fixture registration",
            family=:local,
        )

        @test local_key in available_models(family=:local, shipped=false)
        local_tok = load_tokenizer(local_key; prefetch=false)
        @test local_tok isa WordPieceTokenizer
        @test !isempty(tokenize(local_tok, "hello keemena"))

        @test isempty(download_hf_files("org/repo", String[]; outdir=mktempdir()))

        if get(ENV, "KEEMENA_RUN_NETWORK_TESTS", "0") == "1"
            files = download_hf_files(
                "bert-base-uncased",
                ["vocab.txt"];
                revision="main",
                outdir=mktempdir(),
                force=true,
            )
            @test length(files) == 1
            @test isfile(files[1])
        end
    end

    @testset "Section 12 installable gated model flow" begin
        err = try
            install_model!(:llama3_8b_tokenizer)
            nothing
        catch ex
            ex
        end
        @test err isa ArgumentError
        @test occursin("Provide an access token", sprint(showerror, err))

        err2 = try
            install_model!(:llama2_tokenizer; token="")
            nothing
        catch ex
            ex
        end
        @test err2 isa ArgumentError
        @test occursin("gated", lowercase(sprint(showerror, err2)))
    end

    @testset "Section 13 loader contracts and detection" begin
        @test detect_tokenizer_format(fixture("hf_json_wordpiece")) == :hf_tokenizer_json
        @test detect_tokenizer_format(fixture("bpe_gpt2")) == :bpe_gpt2
        @test detect_tokenizer_format(fixture("bpe_encoder")) == :bpe_encoder
        @test detect_tokenizer_format(fixture("sentencepiece", "toy_bpe.model")) == :sentencepiece_model
        @test detect_tokenizer_format(fixture("sentencepiece", "binary_stub.model")) == :sentencepiece_model
        @test detect_tokenizer_format(fixture("tiktoken_model", "tokenizer.model")) == :tiktoken

        files = detect_tokenizer_files(fixture("hf_json_wordpiece"))
        @test files.tokenizer_json !== nothing
        @test files.vocab_json === nothing

        gpt2 = load_bpe_gpt2(
            fixture("bpe_gpt2", "vocab.json"),
            fixture("bpe_gpt2", "merges.txt"),
        )
        @test gpt2 isa ByteBPETokenizer
        @test !isempty(tokenize(gpt2, "hello world"))

        enc = load_bpe_encoder(
            fixture("bpe_encoder", "encoder.json"),
            fixture("bpe_encoder", "vocab.bpe"),
        )
        @test enc isa ByteBPETokenizer
        @test !isempty(tokenize(enc, "hello world"))

        @test load_tiktoken(fixture("tiktoken_model", "tokenizer.model")) isa TiktokenTokenizer
        auto_tiktoken = load_tokenizer(fixture("tiktoken_model", "tokenizer.model"))
        @test auto_tiktoken isa TiktokenTokenizer

        msg = try
            load_bpe_gpt2(fixture("bpe_gpt2", "vocab.json"), fixture("bpe_gpt2", "missing.txt"))
            nothing
        catch ex
            sprint(showerror, ex)
        end
        @test msg isa String
        @test occursin("Expected files", msg)
        @test occursin("load_bpe_gpt2", msg)

        msg2 = try
            load_tokenizer(fixture("tiktoken_model", "tokenizer.model"); format=:sentencepiece_model)
            nothing
        catch ex
            sprint(showerror, ex)
        end
        @test msg2 isa String
        @test occursin("tiktoken", lowercase(msg2))

        msg3 = try
            load_tokenizer(fixture("sentencepiece", "toy_bpe.model"); format=:tiktoken)
            nothing
        catch ex
            sprint(showerror, ex)
        end
        @test msg3 isa String
        @test occursin("tiktoken", lowercase(msg3))

        local_auto = :local_auto_wordpiece
        register_local_model!(
            local_auto,
            fixture("wordpiece");
            format=:auto,
            description="Auto-detected local wordpiece",
        )
        @test load_tokenizer(local_auto; prefetch=false) isa WordPieceTokenizer

        local_spec = :local_spec_bpe
        register_local_model!(
            local_spec,
            (
                format=:bpe_gpt2,
                vocab_json=fixture("bpe_gpt2", "vocab.json"),
                merges_txt=fixture("bpe_gpt2", "merges.txt"),
            );
            description="Spec-registered local bpe model",
            family=:local,
            notes="section13-spec",
        )
        spec_tok = load_tokenizer(local_spec; prefetch=false)
        @test spec_tok isa ByteBPETokenizer
        spec_info = describe_model(local_spec)
        @test !isempty(spec_info.resolved_files)
        @test occursin("section13-spec", spec_info.notes)
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
