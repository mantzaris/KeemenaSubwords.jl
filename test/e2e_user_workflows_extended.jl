@testset "Section 21 E2E user workflows extended" begin
    corpus = load_edge_case_corpus(; include_long=true)
    categories = load_edge_case_corpus_categories(; include_long=true)
    @test length(corpus) >= 100
    @test haskey(categories, "ascii")
    @test haskey(categories, "unicode")
    @test haskey(categories, "special_tokens")
    @test haskey(categories, "long_generated")

    general_subset = edge_case_corpus_subset(24; include_long=false, nonempty_only=true)
    @test length(general_subset) >= 20

    long_inputs = categories["long_generated"]
    @test length(long_inputs) >= 2
    @test all(length(s) >= 8192 for s in long_inputs)

    @testset "One-call API across families on shared corpus subset" begin
        scenarios = (
            (label="core_bpe_en", source=:core_bpe_en, format=nothing),
            (label="core_wordpiece_en", source=:core_wordpiece_en, format=nothing),
            (label="core_sentencepiece_unigram_en", source=:core_sentencepiece_unigram_en, format=nothing),
            (label="hf_json_wordpiece_fixture", source=fixture("hf_json_wordpiece"), format=:hf_tokenizer_json),
            (label="hf_json_bytelevel_fixture", source=fixture("hf_json_bytelevel_bpe"), format=:hf_tokenizer_json),
            (label="wordpiece_fixture", source=fixture("wordpiece", "vocab.txt"), format=:wordpiece_vocab),
            (label="tiktoken_fixture", source=fixture("tiktoken_model", "tokenizer.model"), format=:tiktoken),
        )

        clear_tokenizer_cache!()
        for scenario in scenarios
            @testset "$(scenario.label)" begin
                texts = scenario.label == "tiktoken_fixture" ?
                    ["hi", "hello", "hi hi", "hello hi", "hi hello"] :
                    general_subset

                for text in texts
                    ids = scenario.format === nothing ?
                        encode(scenario.source, text; add_special_tokens=false) :
                        encode(scenario.source, text; format=scenario.format, add_special_tokens=false)
                    @test ids isa Vector{Int}
                    any(!isspace, text) && @test !isempty(ids)

                    tokens = scenario.format === nothing ?
                        tokenize(scenario.source, text) :
                        tokenize(scenario.source, text; format=scenario.format)
                    @test tokens isa Vector{String}

                    decoded = scenario.format === nothing ?
                        decode(scenario.source, ids) :
                        decode(scenario.source, ids; format=scenario.format)
                    @test decoded isa String

                    result = scenario.format === nothing ?
                        encode_result(scenario.source, text; add_special_tokens=false, return_offsets=true, return_masks=true) :
                        encode_result(scenario.source, text; format=scenario.format, add_special_tokens=false, return_offsets=true, return_masks=true)
                    @test result.ids == ids
                    @test result.attention_mask !== nothing
                    @test result.token_type_ids !== nothing
                    @test result.special_tokens_mask !== nothing
                end
            end
        end
        @test !isempty(cached_tokenizers())
    end

    @testset "HF tokenizer.json realistic added-tokens and special-token workflow" begin
        tok = load_hf_tokenizer_json(fixture("hf_json_realistic_pipeline", "tokenizer.json"))
        @test tok isa HuggingFaceJSONTokenizer

        ids_city = encode(tok, "I love   <city>   !"; add_special_tokens=false)
        ids_city_center = encode(tok, "I love <city_center> !"; add_special_tokens=false)
        @test token_to_id(tok, "<CITY>") in ids_city
        @test token_to_id(tok, "<CITY_CENTER>") in ids_city_center
        @test !(token_to_id(tok, "<CITY>") in ids_city_center) # longest match should win

        # single_word=true should prevent inline matches.
        inline_ids = encode(tok, "prefix<city>suffix"; add_special_tokens=false)
        @test !(token_to_id(tok, "<CITY>") in inline_ids)

        # special token should bypass normalizer and match verbatim.
        special_ids = encode(tok, "[SPECIAL] i"; add_special_tokens=false)
        @test token_to_id(tok, "[SPECIAL]") == first(special_ids)

        # With special tokens enabled, template should insert [CLS] and [SEP].
        templated = encode(tok, "I love city"; add_special_tokens=true)
        @test first(templated) == token_to_id(tok, "[CLS]")
        @test last(templated) == token_to_id(tok, "[SEP]")

        result = encode_result(tok, "I love city"; add_special_tokens=true, return_masks=true)
        @test result.ids == templated
        @test result.special_tokens_mask !== nothing
        @test first(result.special_tokens_mask) == 1
        @test last(result.special_tokens_mask) == 1
    end

    @testset "Local registry workflow over corpus subset" begin
        local_key = Symbol("e2e_section21_local_hf_" * string(time_ns()))
        register_local_model!(
            local_key,
            fixture("hf_json_wordpiece");
            format=:hf_tokenizer_json,
            family=:local,
            description="Section 21 local corpus workflow",
        )

        sample = edge_case_corpus_subset(12; include_long=false, nonempty_only=true)
        for text in sample
            ids = encode(local_key, text; prefetch=false, add_special_tokens=false)
            @test ids isa Vector{Int}
            any(!isspace, text) && @test !isempty(ids)
            @test decode(local_key, ids; prefetch=false) isa String
        end
    end

    @testset "Long-input smoke for selected families" begin
        long_a = long_inputs[1]
        long_b = long_inputs[2]
        ids_wp = encode(:core_wordpiece_en, long_a; add_special_tokens=false)
        ids_hf = encode(fixture("hf_json_bytelevel_bpe"), long_b; format=:hf_tokenizer_json, add_special_tokens=false)
        @test !isempty(ids_wp)
        @test !isempty(ids_hf)
    end

    if _DOWNLOAD_TESTS
        @testset "Download-enabled realistic corpus subset + cache reuse" begin
            keys = [
                :tiktoken_o200k_base,
                :openai_gpt2_bpe,
                :bert_base_uncased_wordpiece,
                :t5_small_sentencepiece_unigram,
                :qwen2_5_bpe,
            ]
            texts = edge_case_corpus_subset(8; include_long=false, nonempty_only=true)
            status_first = prefetch_models_status(keys)
            clear_tokenizer_cache!()

            for key in keys
                st = status_first[key]
                @test st.available
                @test st.method in (:artifact, :fallback_download, :already_present)

                for text in texts
                    result = if key in (:tiktoken_o200k_base, :tiktoken_cl100k_base)
                        encode_result(key, text; add_special_tokens=false, return_masks=true)
                    else
                        encode_result(key, text; return_masks=true)
                    end
                    @test !isempty(result.ids)
                    @test result.attention_mask !== nothing
                    @test decode(key, result.ids) isa String
                end
            end

            cached_after_first = cached_tokenizers()
            @test count(k -> k[1] == :model_key, cached_after_first) >= length(keys)

            # A second prefetch pass should resolve to available entries without failures.
            status_second = prefetch_models_status(keys)
            for key in keys
                @test status_second[key].available
                @test status_second[key].method in (:artifact, :fallback_download, :already_present)
            end

            # Repeat one-call encode to exercise cache reuse path.
            for key in keys
                text = texts[1]
                if key in (:tiktoken_o200k_base, :tiktoken_cl100k_base)
                    @test !isempty(encode(key, text; add_special_tokens=false))
                else
                    @test !isempty(encode(key, text))
                end
            end
            @test length(cached_tokenizers()) >= length(cached_after_first)
        end
    end
end
