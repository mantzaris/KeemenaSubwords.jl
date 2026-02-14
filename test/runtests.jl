using Test
using KeemenaSubwords

const FIXTURES_DIR = joinpath(@__DIR__, "fixtures")
fixture(parts...) = joinpath(FIXTURES_DIR, parts...)
const _DOWNLOAD_TESTS = get(ENV, "KEEMENA_TEST_DOWNLOADS", get(ENV, "KEEMENA_RUN_NETWORK_TESTS", "0")) == "1"
include("helpers/corpus.jl")

function _assert_offset_contract(
    text::String,
    offsets::Vector{Tuple{Int,Int}},
)::Nothing
    base = offsets_index_base()
    max_stop = ncodeunits(text) + base
    prev_start = base
    prev_stop = base
    seen_spanful = false

    for offset in offsets
        if !has_span(offset)
            @test offset == offsets_sentinel()
            continue
        end

        start_idx, stop_idx = offset
        @test base <= start_idx <= stop_idx <= max_stop

        if seen_spanful
            @test start_idx >= prev_start
            @test stop_idx >= prev_stop
        end

        prev_start = start_idx
        prev_stop = stop_idx
        seen_spanful = true
    end

    return nothing
end

function _span_text(text::String, offset::Tuple{Int,Int})::String
    has_span(offset) || return ""
    start_idx, stop_idx = offset
    stop_idx > start_idx || return ""
    return String(SubString(text, start_idx, prevind(text, stop_idx)))
end

@testset "KeemenaSubwords sections 1-21" begin
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
        @test "vocab.json + merges.txt" in gpt2_info.expected_files
        @test "encoder.json + vocab.bpe" in gpt2_info.expected_files
        @test !isempty(gpt2_info.provenance_urls)

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

        hf_realistic = load_hf_tokenizer_json(fixture("hf_json_realistic_pipeline", "tokenizer.json"))
        @test hf_realistic isa HuggingFaceJSONTokenizer

        # Added-token overlap should prefer the longest matching token.
        ids_overlap = encode(hf_realistic, "I love <city_center> !"; add_special_tokens=false)
        @test token_to_id(hf_realistic, "<CITY_CENTER>") in ids_overlap
        @test !(token_to_id(hf_realistic, "<CITY>") in ids_overlap)

        # Added-token single_word should reject embedded matches.
        ids_inline = encode(hf_realistic, "prefix<city>suffix"; add_special_tokens=false)
        @test !(token_to_id(hf_realistic, "<CITY>") in ids_inline)

        # Special tokens bypass normalization and match verbatim.
        ids_special_case = encode(hf_realistic, "[SPECIAL] i"; add_special_tokens=false)
        @test token_to_id(hf_realistic, "[SPECIAL]") == first(ids_special_case)
        ids_not_special = encode(hf_realistic, "[special] i"; add_special_tokens=false)
        @test token_to_id(hf_realistic, "[SPECIAL]") != first(ids_not_special)

        # TemplateProcessing should insert [CLS]/[SEP] in single sequence mode.
        templated_realistic = encode(hf_realistic, "I love city"; add_special_tokens=true)
        @test first(templated_realistic) == token_to_id(hf_realistic, "[CLS]")
        @test last(templated_realistic) == token_to_id(hf_realistic, "[SEP]")

        hf_fallback = load_hf_tokenizer_json(fixture("hf_json_byte_fallback", "tokenizer.json"))
        @test hf_fallback isa HuggingFaceJSONTokenizer

        # Byte fallback should avoid UNK for unseen bytes when byte tokens exist.
        euro_ids = encode(hf_fallback, "€"; add_special_tokens=false)
        @test !isempty(euro_ids)
        @test !(token_to_id(hf_fallback, "<unk>") in euro_ids)

        emoji_ids = encode(hf_fallback, "😀"; add_special_tokens=false)
        @test !isempty(emoji_ids)
        @test !(token_to_id(hf_fallback, "<unk>") in emoji_ids)

        # Run representative corpus subset through multiple HF fixtures.
        hf_subset = edge_case_corpus_subset(18; include_long=false, nonempty_only=true)
        @test length(hf_subset) >= 12
        for text in hf_subset
            @test encode(hf_bpe, text; add_special_tokens=false) isa Vector{Int}
            @test decode(hf_bpe, encode(hf_bpe, text; add_special_tokens=false)) isa String
            @test encode(hf_realistic, text; add_special_tokens=false) isa Vector{Int}
            @test decode(hf_realistic, encode(hf_realistic, text; add_special_tokens=false)) isa String
        end
    end

    @testset "Section 9 built-in public baseline keys" begin
        baseline_keys = [
            :tiktoken_o200k_base,
            :tiktoken_cl100k_base,
            :tiktoken_r50k_base,
            :tiktoken_p50k_base,
            :openai_gpt2_bpe,
            :bert_base_uncased_wordpiece,
            :bert_base_multilingual_cased_wordpiece,
            :t5_small_sentencepiece_unigram,
        ]
        for key in baseline_keys
            info = describe_model(key)
            @test info.distribution == :artifact_public
            @test !isempty(info.expected_files)
            @test !isempty(info.provenance_urls)
        end
    end

    @testset "Section 9 curated model inventory keys" begin
        curated_keys = [
            :mistral_v1_sentencepiece,
            :mistral_v3_sentencepiece,
            :phi2_bpe,
            :qwen2_5_bpe,
            :roberta_base_bpe,
            :xlm_roberta_base_sentencepiece_bpe,
        ]
        for key in curated_keys
            info = describe_model(key)
            @test info.distribution == :artifact_public
            @test !isempty(info.expected_files)
            @test !isempty(info.provenance_urls)
        end
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

        if _DOWNLOAD_TESTS
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

    include("training/runtests_training.jl")
    include("hf_json_export_roundtrip.jl")
    include("hf_json_bert_components.jl")

    @testset "Section 16 docs contracts and loader coherence" begin
        wp_path = fixture("wordpiece", "vocab.txt")
        wp_by_path = load_tokenizer((format=:wordpiece_vocab, path=wp_path))
        wp_by_alias = load_tokenizer((format=:wordpiece_vocab, vocab_txt=wp_path))
        @test wp_by_path isa WordPieceTokenizer
        @test wp_by_alias isa WordPieceTokenizer
        @test tokenize(wp_by_path, "hello keemena") == tokenize(wp_by_alias, "hello keemena")

        hf_path = fixture("hf_json_wordpiece", "tokenizer.json")
        hf_by_path = load_tokenizer((format=:hf_tokenizer_json, path=hf_path))
        hf_by_alias = load_tokenizer((format=:hf_tokenizer_json, tokenizer_json=hf_path))
        @test hf_by_path isa HuggingFaceJSONTokenizer
        @test hf_by_alias isa HuggingFaceJSONTokenizer
        @test tokenize(hf_by_path, "Hello world") == tokenize(hf_by_alias, "Hello world")

        uni_path = fixture("unigram", "unigram.tsv")
        uni_by_path = load_tokenizer((format=:unigram, path=uni_path))
        uni_by_alias = load_tokenizer((format=:unigram, unigram_tsv=uni_path))
        @test uni_by_path isa UnigramTokenizer
        @test uni_by_alias isa UnigramTokenizer
        @test tokenize(uni_by_path, "hello world") == tokenize(uni_by_alias, "hello world")

        combined = mktempdir()
        cp(fixture("hf_json_wordpiece", "tokenizer.json"), joinpath(combined, "tokenizer.json"))
        cp(fixture("bpe_gpt2", "vocab.json"), joinpath(combined, "vocab.json"))
        cp(fixture("bpe_gpt2", "merges.txt"), joinpath(combined, "merges.txt"))

        @test detect_tokenizer_format(combined) == :hf_tokenizer_json
        @test load_tokenizer(combined) isa HuggingFaceJSONTokenizer
        @test load_tokenizer(combined; format=:bpe_gpt2) isa ByteBPETokenizer

        @test load_bpe(fixture("bpe")) isa BPETokenizer
        @test load_bytebpe(fixture("bpe")) isa ByteBPETokenizer
        @test load_unigram(fixture("unigram")) isa UnigramTokenizer
    end

    @testset "Section 17 structured outputs and file specs" begin
        wp = load_tokenizer(:core_wordpiece_en)
        result = encode_result(wp, "hello world"; add_special_tokens=false, return_offsets=true, return_masks=true)
        @test result isa TokenizationResult
        @test result.ids == encode(wp, "hello world"; add_special_tokens=false)
        @test result.tokens == tokenize(wp, "hello world")
        @test result.offsets !== nothing
        @test length(result.offsets) == length(result.ids)
        @test result.attention_mask == fill(1, length(result.ids))
        @test result.token_type_ids == fill(0, length(result.ids))
        @test result.special_tokens_mask == fill(0, length(result.ids))
        @test result.metadata.format == :wordpiece

        hf = load_tokenizer(fixture("hf_json_wordpiece"); format=:hf_tokenizer_json)
        hf_result = encode_result(hf, "hello world"; add_special_tokens=true, return_offsets=true, return_masks=true)
        @test hf_result.ids == encode(hf, "hello world"; add_special_tokens=true)
        @test hf_result.offsets !== nothing
        @test length(hf_result.offsets) == length(hf_result.ids)
        @test hf_result.attention_mask == fill(1, length(hf_result.ids))
        @test length(hf_result.special_tokens_mask) == length(hf_result.ids)
        @test sum(hf_result.special_tokens_mask) >= 2
        for (i, mask) in enumerate(hf_result.special_tokens_mask)
            if mask == 1
                @test hf_result.offsets[i] == (0, 0)
            end
        end

        batch = encode_batch_result(wp, ["hello world", "hello keemena"]; return_masks=true)
        @test length(batch) == 2
        @test batch[1] isa TokenizationResult

        hf_added = load_hf_tokenizer_json(fixture("hf_json_added_tokens", "tokenizer.json"))
        city_id = token_to_id(hf_added, "<CITY>")
        @test city_id in encode(hf_added, "I love <CITY> !")
        @test !(city_id in encode(hf_added, "xx<CITY>yy"))  # single_word boundary should block match

        spec = FilesSpec(
            format=:bpe_gpt2,
            vocab_json=fixture("bpe_gpt2", "vocab.json"),
            merges_txt=fixture("bpe_gpt2", "merges.txt"),
        )
        tok = load_tokenizer(spec)
        @test tok isa ByteBPETokenizer

        local_key = :local_filespec_bpe
        register_local_model!(
            local_key,
            spec;
            description="FilesSpec local registration",
            family=:local,
        )
        loaded = load_tokenizer(local_key; prefetch=false)
        @test loaded isa ByteBPETokenizer

        # Additional precedence hardening in mixed directories.
        mixed = mktempdir()
        cp(fixture("bpe_encoder", "encoder.json"), joinpath(mixed, "encoder.json"))
        cp(fixture("bpe_encoder", "vocab.bpe"), joinpath(mixed, "vocab.bpe"))
        cp(fixture("bpe_gpt2", "vocab.json"), joinpath(mixed, "vocab.json"))
        cp(fixture("bpe_gpt2", "merges.txt"), joinpath(mixed, "merges.txt"))
        @test detect_tokenizer_format(mixed) == :bpe_gpt2
    end

    @testset "Section 20 asset UX, locking, and one-call APIs" begin
        status = prefetch_models_status([:core_bpe_en])
        @test haskey(status, :core_bpe_en)
        @test hasproperty(status[:core_bpe_en], :available)
        @test hasproperty(status[:core_bpe_en], :method)
        @test hasproperty(status[:core_bpe_en], :path)
        @test hasproperty(status[:core_bpe_en], :error)
        @test status[:core_bpe_en].available
        @test status[:core_bpe_en].method in (:artifact, :already_present, :fallback_download)

        compat = prefetch_models([:core_bpe_en])
        @test compat[:core_bpe_en] == status[:core_bpe_en].available
        @test asset_status(:core_bpe_en).available == status[:core_bpe_en].available

        spiece_dir = mktempdir()
        cp(fixture("sentencepiece", "toy_bpe.model"), joinpath(spiece_dir, "spiece.model"))
        detected_spiece = detect_tokenizer_files(spiece_dir)
        @test length(detected_spiece.sentencepiece_models) == 1
        @test basename(only(detected_spiece.sentencepiece_models)) == "spiece.model"
        @test detect_tokenizer_format(spiece_dir) == :sentencepiece_model
        @test load_tokenizer(spiece_dir) isa SentencePieceTokenizer

        custom_model_dir = mktempdir()
        cp(fixture("sentencepiece", "toy_bpe.model"), joinpath(custom_model_dir, "custom.model"))
        detected_custom = detect_tokenizer_files(custom_model_dir)
        @test length(detected_custom.sentencepiece_models) == 1
        @test basename(only(detected_custom.sentencepiece_models)) == "custom.model"
        @test detect_tokenizer_format(custom_model_dir) == :sentencepiece_model
        @test load_tokenizer(custom_model_dir) isa SentencePieceTokenizer

        clear_tokenizer_cache!()
        @test encode(:core_bpe_en, "hello world") == encode(load_tokenizer(:core_bpe_en), "hello world")
        @test !isempty(tokenize(fixture("hf_json_wordpiece"), "Hello world"))
        @test encode_result(fixture("hf_json_wordpiece"), "Hello world").ids == encode(
            load_tokenizer(fixture("hf_json_wordpiece")),
            "Hello world";
            add_special_tokens=true,
        )
        @test decode(:core_wordpiece_en, encode(:core_wordpiece_en, "hello world"; add_special_tokens=true)) == "hello world"
        @test !isempty(cached_tokenizers())
        clear_tokenizer_cache!()
        @test isempty(cached_tokenizers())

        lock_key = Symbol("section20_lock_" * string(time_ns()))
        t1_end = Ref(0.0)
        t2_start = Ref(0.0)
        t1 = Threads.@spawn KeemenaSubwords._with_model_lock(lock_key; timeout_sec=5.0) do
            sleep(0.20)
            t1_end[] = time()
        end
        sleep(0.02)
        t2 = Threads.@spawn KeemenaSubwords._with_model_lock(lock_key; timeout_sec=5.0) do
            t2_start[] = time()
        end
        fetch(t1)
        fetch(t2)
        @test t2_start[] >= t1_end[] - 1e-3

        @test_throws ErrorException KeemenaSubwords._with_model_lock(lock_key; timeout_sec=5.0) do
            error("lock release validation")
        end
        @test KeemenaSubwords._with_model_lock(lock_key; timeout_sec=5.0) do
            true
        end

        if _DOWNLOAD_TESTS
            smoke_keys = [
                :tiktoken_o200k_base,
                :openai_gpt2_bpe,
                :bert_base_uncased_wordpiece,
                :t5_small_sentencepiece_unigram,
                :qwen2_5_bpe,
            ]
            smoke = prefetch_models_status(smoke_keys)
            for key in smoke_keys
                @test haskey(smoke, key)
                entry = smoke[key]
                @test entry.available
                @test entry.method in (:artifact, :fallback_download, :already_present)
                @test entry.path !== nothing

                tok = load_tokenizer(key)
                ids = encode(tok, "Hello world")
                @test !isempty(ids)
                e2e = if tok isa TiktokenTokenizer
                    encode_result(key, "Hello world"; add_special_tokens=false, return_masks=true)
                else
                    encode_result(key, "Hello world"; return_masks=true)
                end
                @test !isempty(e2e.ids)
                @test e2e.attention_mask !== nothing
            end
        end
    end

    @testset "Section 21 shared edge-case corpus" begin
        corpus = load_edge_case_corpus(; include_long=true)
        @test length(corpus) >= 100
        @test any(s -> any(!isascii, s), corpus)
        @test any(s -> occursin('\n', s), corpus)
        @test any(s -> occursin("😀", s), corpus)
        @test any(s -> length(s) >= 8192, corpus)
    end

    @testset "Section 22 offset contract invariants" begin
        @test offsets_coordinate_system() == :utf8_codeunits
        @test offsets_index_base() == 1
        @test offsets_span_style() == :half_open
        @test offsets_sentinel() == (0, 0)
        @test !has_span(offsets_sentinel())
        @test has_span((1, 1))

        @testset "Normalization bypass correctness" begin
            tok = load_hf_tokenizer_json(fixture("hf_json_wordpiece", "tokenizer.json"))
            input = "ＨELLO WORLD"
            normalized = normalize(tok, input)

            direct = encode_result(
                tok,
                input;
                add_special_tokens=false,
                assume_normalized=false,
                return_offsets=true,
                return_masks=true,
            )
            bypass = encode_result(
                tok,
                normalized;
                add_special_tokens=false,
                assume_normalized=true,
                return_offsets=true,
                return_masks=true,
            )

            @test direct.ids == bypass.ids
            @test bypass.offsets !== nothing
            _assert_offset_contract(normalized, bypass.offsets)
        end

        @testset "Inserted versus present-in-text special spans" begin
            tok = load_hf_tokenizer_json(fixture("hf_json_special_spans", "tokenizer.json"))
            normalized = normalize(tok, "[SPECIAL] i")
            result = encode_result(
                tok,
                normalized;
                assume_normalized=true,
                add_special_tokens=true,
                return_offsets=true,
                return_masks=true,
            )

            @test result.offsets !== nothing
            @test result.special_tokens_mask !== nothing
            @test length(result.offsets) == length(result.ids)
            _assert_offset_contract(normalized, result.offsets)

            cls_idx = findfirst(==("[CLS]"), result.tokens)
            sep_idx = findlast(==("[SEP]"), result.tokens)
            present_idx = findfirst(==("[SPECIAL]"), result.tokens)

            @test cls_idx !== nothing
            @test sep_idx !== nothing
            @test present_idx !== nothing

            @test result.special_tokens_mask[cls_idx] == 1
            @test result.special_tokens_mask[sep_idx] == 1
            @test result.offsets[cls_idx] == offsets_sentinel()
            @test result.offsets[sep_idx] == offsets_sentinel()

            @test result.special_tokens_mask[present_idx] == 1
            @test has_span(result.offsets[present_idx])
            @test _span_text(normalized, result.offsets[present_idx]) == "[SPECIAL]"
        end

        @testset "Cross-family bounds, sentinel, and monotonicity" begin
            cases = (
                (
                    label="tiktoken",
                    tokenizer=load_tiktoken(fixture("tiktoken_model", "tokenizer.model")),
                    add_special=false,
                    texts=["hi", "hello", "hi hi", "hello hi", "hi hello"],
                ),
                (
                    label="bytebpe",
                    tokenizer=load_bpe_gpt2(fixture("bpe_gpt2", "vocab.json"), fixture("bpe_gpt2", "merges.txt")),
                    add_special=true,
                    texts=["hello world", "hello  world", "hello\tworld"],
                ),
                (
                    label="wordpiece",
                    tokenizer=load_wordpiece(fixture("wordpiece", "vocab.txt")),
                    add_special=true,
                    texts=["hello world", "hello", "hello keemena"],
                ),
                (
                    label="sentencepiece_unigram",
                    tokenizer=load_sentencepiece(model_path(:core_sentencepiece_unigram_en)),
                    add_special=true,
                    texts=["hello world", "hello", "world"],
                ),
                (
                    label="sentencepiece_bpe",
                    tokenizer=load_sentencepiece(fixture("sentencepiece", "toy_bpe.model")),
                    add_special=true,
                    texts=["hello world", "hello", "world"],
                ),
                (
                    label="unigram_tsv",
                    tokenizer=load_unigram(fixture("unigram", "unigram.tsv")),
                    add_special=true,
                    texts=["hello world", "hello", "world"],
                ),
                (
                    label="hf_wordpiece_json",
                    tokenizer=load_hf_tokenizer_json(fixture("hf_json_wordpiece", "tokenizer.json")),
                    add_special=true,
                    texts=["hello world", "hello", "hello world"],
                ),
                (
                    label="hf_bpe_json",
                    tokenizer=load_hf_tokenizer_json(fixture("hf_json_bytelevel_bpe", "tokenizer.json")),
                    add_special=true,
                    texts=["hello world", " hello world", "hello  world"],
                ),
                (
                    label="hf_unigram_json",
                    tokenizer=load_hf_tokenizer_json(fixture("hf_json_unigram_metaspace", "tokenizer.json")),
                    add_special=true,
                    texts=["hello world", "hello", "world"],
                ),
            )

            for case in cases
                @testset "$(case.label)" begin
                    for text in case.texts
                        normalized = normalize(case.tokenizer, text)
                        result = encode_result(
                            case.tokenizer,
                            normalized;
                            add_special_tokens=case.add_special,
                            assume_normalized=true,
                            return_offsets=true,
                            return_masks=true,
                        )

                        @test result.offsets !== nothing
                        @test length(result.ids) == length(result.tokens)
                        @test length(result.offsets) == length(result.ids)
                        @test length(result.attention_mask) == length(result.ids)
                        @test length(result.token_type_ids) == length(result.ids)
                        @test length(result.special_tokens_mask) == length(result.ids)

                        for offset in result.offsets
                            @test has_span(offset) == (offset != offsets_sentinel())
                        end
                        _assert_offset_contract(normalized, result.offsets)
                    end
                end
            end
        end

        @testset "1-based regression guard" begin
            wp = load_wordpiece(fixture("wordpiece", "vocab.txt"))
            result = encode_result(
                wp,
                "hello";
                add_special_tokens=false,
                assume_normalized=true,
                return_offsets=true,
                return_masks=true,
            )

            @test result.offsets !== nothing
            first_span_idx = findfirst(has_span, result.offsets)
            @test first_span_idx !== nothing
            @test result.offsets[first_span_idx][1] == offsets_index_base()
        end

        @testset "ByteLevel and whitespace-sensitive offsets" begin
            tok = load_hf_tokenizer_json(fixture("hf_json_bytelevel_bpe", "tokenizer.json"))
            whitespace_cases = [
                "hello world",
                " hello world",
                "hello  world",
                "hello\tworld",
                "hello\nworld",
                "  hello\tworld\n",
            ]

            for text in whitespace_cases
                normalized = normalize(tok, text)
                result = encode_result(
                    tok,
                    normalized;
                    assume_normalized=true,
                    add_special_tokens=false,
                    return_offsets=true,
                    return_masks=true,
                )

                @test result.offsets !== nothing
                _assert_offset_contract(normalized, result.offsets)
            end

            prefix_space_case = normalize(tok, "hello world")
            prefix_space_result = encode_result(
                tok,
                prefix_space_case;
                assume_normalized=true,
                add_special_tokens=false,
                return_offsets=true,
                return_masks=true,
            )
            @test prefix_space_result.offsets !== nothing
            first_span_idx = findfirst(has_span, prefix_space_result.offsets)
            @test first_span_idx !== nothing
            @test prefix_space_result.offsets[first_span_idx][1] == offsets_index_base()
        end
    end

    @testset "Section 23 offsets robustness and downstream-safe span utilities" begin
        @testset "Helper semantics: sentinel, empty, non-empty" begin
            sentinel = offsets_sentinel()
            @test !has_span(sentinel)
            @test !has_nonempty_span(sentinel)

            empty = (1, 1)
            @test has_span(empty)
            @test !has_nonempty_span(empty)

            nonempty = (1, 2)
            @test has_span(nonempty)
            @test has_nonempty_span(nonempty)

            @test span_ncodeunits(sentinel) == 0
            @test span_ncodeunits(empty) == 0
            @test span_ncodeunits(nonempty) == 1

            @test span_codeunits("hello", sentinel) == UInt8[]
            @test span_codeunits("hello", empty) == UInt8[]
            @test span_codeunits("hello", (1, 3)) == Vector{UInt8}(codeunits("he"))

            sample = "a🙂b"
            @test is_valid_string_boundary(sample, 1)
            @test is_valid_string_boundary(sample, 2)
            @test is_valid_string_boundary(sample, ncodeunits(sample) + 1)
            @test !is_valid_string_boundary(sample, 0)
            @test !is_valid_string_boundary(sample, 3)
            @test !is_valid_string_boundary(sample, ncodeunits(sample) + 2)

            @test_nowarn try_span_substring(sample, sentinel)
            @test_nowarn try_span_substring(sample, empty)
            @test_nowarn try_span_substring(sample, (2, 6))
            @test_nowarn try_span_substring(sample, (3, 6))
            @test try_span_substring(sample, sentinel) == ""
            @test try_span_substring(sample, empty) == ""
            @test try_span_substring(sample, (1, 2)) == "a"
            @test try_span_substring(sample, (2, 6)) == "🙂"
            @test try_span_substring(sample, (3, 6)) === nothing
        end

        @testset "Helper semantics: non-overlap predicate" begin
            @test offsets_are_nonoverlapping([(1, 2), (2, 3), (3, 3), (3, 4)])
            @test !offsets_are_nonoverlapping([(1, 3), (2, 4)])
            @test offsets_are_nonoverlapping([(0, 0), (1, 2), (2, 2), (2, 3)])
            @test !offsets_are_nonoverlapping([(0, 0), (1, 2)]; ignore_sentinel=false)
        end

        @testset "Tokenizer non-overlap regression checks" begin
            cases = (
                (
                    label="wordpiece",
                    tokenizer=load_wordpiece(fixture("wordpiece", "vocab.txt")),
                    text="hello world",
                    add_special=true,
                ),
                (
                    label="sentencepiece_unigram",
                    tokenizer=load_sentencepiece(model_path(:core_sentencepiece_unigram_en)),
                    text="hello world",
                    add_special=true,
                ),
                (
                    label="hf_wordpiece_json",
                    tokenizer=load_hf_tokenizer_json(fixture("hf_json_wordpiece", "tokenizer.json")),
                    text="hello world",
                    add_special=true,
                ),
                (
                    label="classic_bpe",
                    tokenizer=load_tokenizer(:core_bpe_en),
                    text="hello world",
                    add_special=true,
                ),
                (
                    label="bytebpe",
                    tokenizer=load_bpe_gpt2(fixture("bpe_gpt2", "vocab.json"), fixture("bpe_gpt2", "merges.txt")),
                    text="hello world",
                    add_special=true,
                ),
                (
                    label="tiktoken",
                    tokenizer=load_tiktoken(fixture("tiktoken_model", "tokenizer.model")),
                    text="hi hello",
                    add_special=false,
                ),
            )

            for case in cases
                @testset "$(case.label)" begin
                    normalized = normalize(case.tokenizer, case.text)
                    result = encode_result(
                        case.tokenizer,
                        normalized;
                        add_special_tokens=case.add_special,
                        assume_normalized=true,
                        return_offsets=true,
                        return_masks=true,
                    )

                    @test result.offsets !== nothing
                    _assert_offset_contract(normalized, result.offsets)
                    @test offsets_are_nonoverlapping(
                        result.offsets;
                        ignore_sentinel=true,
                        ignore_empty=true,
                    )
                end
            end
        end

        @testset "Byte-level multibyte safety helpers" begin
            tokenizers = (
                (
                    label="bytebpe",
                    tokenizer=load_bpe_gpt2(fixture("bpe_gpt2", "vocab.json"), fixture("bpe_gpt2", "merges.txt")),
                ),
                (
                    label="hf_bytelevel",
                    tokenizer=load_hf_tokenizer_json(fixture("hf_json_bytelevel_bpe", "tokenizer.json")),
                ),
            )
            texts = ["cafe\u0301", "é", "🙂", "a🙂b"]

            for spec in tokenizers
                @testset "$(spec.label)" begin
                    for text in texts
                        normalized = normalize(spec.tokenizer, text)
                        result = encode_result(
                            spec.tokenizer,
                            normalized;
                            assume_normalized=true,
                            add_special_tokens=false,
                            return_offsets=true,
                            return_masks=true,
                        )

                        @test result.offsets !== nothing
                        _assert_offset_contract(normalized, result.offsets)
                        @test offsets_are_nonoverlapping(
                            result.offsets;
                            ignore_sentinel=true,
                            ignore_empty=true,
                        )

                        for offset in result.offsets
                            @test_nowarn try_span_substring(normalized, offset)
                            sub = try_span_substring(normalized, offset)
                            bytes = span_codeunits(normalized, offset)

                            if has_nonempty_span(offset)
                                @test length(bytes) == span_ncodeunits(offset)
                            else
                                @test isempty(bytes)
                                @test sub == ""
                            end

                            if sub === nothing
                                start_idx, stop_idx = offset
                                @test has_nonempty_span(offset)
                                @test !is_valid_string_boundary(normalized, start_idx) ||
                                    !is_valid_string_boundary(normalized, stop_idx)
                            else
                                @test Vector{UInt8}(codeunits(sub)) == bytes
                            end
                        end
                    end
                end
            end
        end
    end

    @testset "Section 24 boundary-valid offsets guarantees by tokenizer family" begin
        @testset "Section 24 boundary-valid offsets for string-level tokenizers" begin
            cases = (
                (
                    label="wordpiece",
                    tokenizer=load_wordpiece(fixture("wordpiece", "vocab.txt")),
                ),
                (
                    label="sentencepiece_unigram",
                    tokenizer=load_sentencepiece(model_path(:core_sentencepiece_unigram_en)),
                ),
                (
                    label="unigram_tsv",
                    tokenizer=load_unigram(fixture("unigram", "unigram.tsv")),
                ),
                (
                    label="hf_wordpiece_json",
                    tokenizer=load_hf_tokenizer_json(fixture("hf_json_wordpiece", "tokenizer.json")),
                ),
                (
                    label="hf_unigram_json",
                    tokenizer=load_hf_tokenizer_json(fixture("hf_json_unigram_metaspace", "tokenizer.json")),
                ),
                (
                    label="sentencepiece_bpe",
                    tokenizer=load_sentencepiece(fixture("sentencepiece", "toy_bpe.model")),
                ),
                (
                    label="classic_bpe",
                    tokenizer=load_tokenizer(:core_bpe_en),
                ),
            )

            texts = ["cafe\u0301", "é", "🙂", "a🙂b"]

            for case in cases
                @testset "$(case.label)" begin
                    for text in texts
                        normalized = normalize(case.tokenizer, text)
                        result = encode_result(
                            case.tokenizer,
                            normalized;
                            assume_normalized=true,
                            add_special_tokens=false,
                            return_offsets=true,
                            return_masks=true,
                        )

                        @test result.offsets !== nothing
                        _assert_offset_contract(normalized, result.offsets)
                        @test offsets_are_nonoverlapping(
                            result.offsets;
                            ignore_sentinel=true,
                            ignore_empty=true,
                        )

                        for offset in result.offsets
                            has_nonempty_span(offset) || continue
                            start_idx, stop_idx = offset
                            @test is_valid_string_boundary(normalized, start_idx)
                            @test is_valid_string_boundary(normalized, stop_idx)
                            sub = try_span_substring(normalized, offset)
                            @test sub isa String
                            @test Vector{UInt8}(codeunits(sub)) == span_codeunits(normalized, offset)
                        end
                    end
                end
            end
        end

        @testset "Section 24 boundary-valid offsets for byte-level tokenizers on ASCII" begin
            cases = (
                (
                    label="bytebpe",
                    tokenizer=load_bpe_gpt2(fixture("bpe_gpt2", "vocab.json"), fixture("bpe_gpt2", "merges.txt")),
                    texts=["hello", "hello world", " hello world", "hello  world", "hello\tworld", "hello\nworld"],
                ),
                (
                    label="hf_bytelevel",
                    tokenizer=load_hf_tokenizer_json(fixture("hf_json_bytelevel_bpe", "tokenizer.json")),
                    texts=["hello", "hello world", " hello world", "hello  world", "hello\tworld", "hello\nworld"],
                ),
                (
                    label="tiktoken",
                    tokenizer=load_tiktoken(fixture("tiktoken_model", "tokenizer.model")),
                    texts=["hello", "hello hello", " hello", "hello  hello", "hi hello"],
                ),
            )

            for spec in cases
                @testset "$(spec.label)" begin
                    for text in spec.texts
                        normalized = normalize(spec.tokenizer, text)
                        result = encode_result(
                            spec.tokenizer,
                            normalized;
                            assume_normalized=true,
                            add_special_tokens=false,
                            return_offsets=true,
                            return_masks=true,
                        )

                        @test result.offsets !== nothing
                        _assert_offset_contract(normalized, result.offsets)
                        @test offsets_are_nonoverlapping(
                            result.offsets;
                            ignore_sentinel=true,
                            ignore_empty=true,
                        )

                        for offset in result.offsets
                            has_nonempty_span(offset) || continue
                            sub = try_span_substring(normalized, offset)
                            @test sub isa String
                            @test Vector{UInt8}(codeunits(sub)) == span_codeunits(normalized, offset)
                        end
                    end
                end
            end
        end
    end

    @testset "Section 25 strict offset validators" begin
        @testset "validate/assert with sentinel and empty spans" begin
            text = "hello"
            offsets = [offsets_sentinel(), (1, 1), (1, 2), (2, 2)]
            @test validate_offsets_contract(text, offsets)
            @test validate_offsets_contract(text, offsets; require_string_boundaries=true)
            @test_nowarn assert_offsets_contract(text, offsets)
            @test_nowarn assert_offsets_contract(text, offsets; require_string_boundaries=true)
        end

        @testset "known-good offsets from tokenizer output" begin
            wp = load_wordpiece(fixture("wordpiece", "vocab.txt"))
            text = normalize(wp, "hello world")
            result = encode_result(
                wp,
                text;
                assume_normalized=true,
                add_special_tokens=false,
                return_offsets=true,
                return_masks=true,
            )
            @test result.offsets !== nothing
            @test validate_offsets_contract(text, result.offsets)
            @test validate_offsets_contract(text, result.offsets; require_string_boundaries=true)
            @test_nowarn assert_offsets_contract(text, result.offsets)
            @test_nowarn assert_offsets_contract(text, result.offsets; require_string_boundaries=true)
        end

        @testset "invalid offsets fail fast" begin
            text = "hello"
            bad_offsets = [(0, 1), (1, 2)]
            @test !validate_offsets_contract(text, bad_offsets)
            @test_throws ArgumentError assert_offsets_contract(text, bad_offsets)
        end

        @testset "require_string_boundaries option" begin
            text = "🙂"
            non_boundary_offsets = [(2, 3)]
            @test validate_offsets_contract(text, non_boundary_offsets)
            @test !validate_offsets_contract(text, non_boundary_offsets; require_string_boundaries=true)
            @test_nowarn assert_offsets_contract(text, non_boundary_offsets)
            @test_throws ArgumentError assert_offsets_contract(
                text,
                non_boundary_offsets;
                require_string_boundaries=true,
            )
        end
    end

    include("e2e_user_workflows.jl")
    include("e2e_user_workflows_extended.jl")
    include("test_goldens.jl")
    include("test_minimal_expected_ids.jl")
end
