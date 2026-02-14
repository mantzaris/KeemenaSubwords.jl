using JSON3

function _json_has_key(obj, key::String)::Bool
    return haskey(obj, key) || haskey(obj, Symbol(key))
end

function _json_get(obj, key::String)
    if haskey(obj, key)
        return obj[key]
    elseif haskey(obj, Symbol(key))
        return obj[Symbol(key)]
    end
    throw(ArgumentError("Missing key '$key' in JSON object"))
end

function _json_is_object(value)::Bool
    return value !== nothing && (value isa AbstractDict || string(typeof(value)) == "JSON3.Object")
end

function _json_is_array(value)::Bool
    return value !== nothing && (value isa AbstractVector || occursin("JSON3.Array", string(typeof(value))))
end

function _assert_exported_hf_json_sanity(tokenizer_json_path::String)::Nothing
    root = JSON3.read(read(tokenizer_json_path, String))
    model = _json_get(root, "model")
    model_type = String(_json_get(model, "type"))

    if model_type == "Unigram"
        vocab_rows = _json_get(model, "vocab")
        @test !isempty(vocab_rows)
        unk_id_zero = Int(_json_get(model, "unk_id"))
        @test 0 <= unk_id_zero < length(vocab_rows)
    else
        vocab = _json_get(model, "vocab")
        ids = sort(collect(Int(id_zero) for (_, id_zero) in pairs(vocab)))
        @test !isempty(ids)
        @test ids == collect(0:(length(ids) - 1))

        if _json_has_key(model, "unk_token")
            unk_token = String(_json_get(model, "unk_token"))
            @test _json_has_key(vocab, unk_token)
            @test Int(_json_get(vocab, unk_token)) in ids
        end
    end

    if _json_has_key(root, "post_processor")
        post = _json_get(root, "post_processor")
        if post !== nothing && _json_is_object(post) && _json_has_key(post, "type")
            if String(_json_get(post, "type")) == "TemplateProcessing"
                @test _json_has_key(post, "special_tokens")
                special_tokens = _json_get(post, "special_tokens")
                @test _json_is_object(special_tokens)
                @test !_json_is_array(special_tokens)

                for key in ("single", "pair")
                    @test _json_has_key(post, key)
                    template_items = _json_get(post, key)
                    @test _json_is_array(template_items)

                    for item in template_items
                        @test _json_is_object(item)
                        @test _json_has_key(item, "SpecialToken") || _json_has_key(item, "Sequence")
                    end
                end
            end
        end
    end

    return nothing
end

function _assert_hf_export_roundtrip(
    tokenizer::AbstractSubwordTokenizer,
    samples::Vector{String};
    add_special_modes::Tuple{Vararg{Bool}}=(false, true),
    require_string_boundaries::Bool=true,
)::NamedTuple{(:explicit, :auto),Tuple{HuggingFaceJSONTokenizer,HuggingFaceJSONTokenizer}}
    outdir = mktempdir()
    export_tokenizer(tokenizer, outdir; format=:hf_tokenizer_json)

    tokenizer_json_path = joinpath(outdir, "tokenizer.json")
    @test isfile(tokenizer_json_path)
    _assert_exported_hf_json_sanity(tokenizer_json_path)

    reloaded = load_tokenizer(outdir; format=:hf_tokenizer_json)
    @test reloaded isa HuggingFaceJSONTokenizer
    reloaded_auto = load_tokenizer(outdir)
    @test reloaded_auto isa HuggingFaceJSONTokenizer

    for candidate in (reloaded, reloaded_auto)
        for text in samples
            @test tokenize(candidate, text) == tokenize(tokenizer, text)

            for add_special_tokens in add_special_modes
                ids_original = encode(tokenizer, text; add_special_tokens=add_special_tokens)
                ids_reloaded = encode(candidate, text; add_special_tokens=add_special_tokens)
                @test ids_reloaded == ids_original
                @test decode(candidate, ids_reloaded) == decode(tokenizer, ids_original)

                tokenization_text = tokenization_view(candidate, text)
                result = encode_result(
                    candidate,
                    tokenization_text;
                    assume_normalized=true,
                    add_special_tokens=add_special_tokens,
                    return_offsets=true,
                    return_masks=true,
                )

                @test result.offsets !== nothing
                @test_nowarn assert_offsets_contract(
                    tokenization_text,
                    result.offsets;
                    require_string_boundaries=require_string_boundaries,
                )
                @test offsets_are_nonoverlapping(
                    result.offsets;
                    ignore_sentinel=true,
                    ignore_empty=true,
                )
            end
        end
    end

    return (explicit=reloaded, auto=reloaded_auto)
end

@testset "Section 26 Hugging Face tokenizer.json export round-trip" begin
    corpus = [
        "hello world",
        "hello, world!",
        "token mix case",
        "subword tokenization stays deterministic",
        "cafe token mix",
        "café token mix",
    ]

    samples = [
        "hello world",
        "hello, world!",
        "café token mix",
    ]

    @testset "BPE export/reload parity" begin
        tokenizer = train_bpe(
            corpus;
            vocab_size=96,
            min_frequency=1,
            special_tokens=Dict(
                :unk => "<UNK>",
                :pad => "<PAD>",
                :bos => "<BOS>",
                :eos => "<EOS>",
            ),
            model_name="hf_json_export_bpe",
            version=v"0.3.0",
        )

        _assert_hf_export_roundtrip(tokenizer, samples)
    end

    @testset "WordPiece export/reload parity" begin
        tokenizer = train_wordpiece(
            corpus;
            vocab_size=128,
            min_frequency=1,
            special_tokens=Dict(
                :unk => "[UNK]",
                :pad => "[PAD]",
                :cls => "[CLS]",
                :sep => "[SEP]",
                :mask => "[MASK]",
            ),
            continuation_prefix="##",
            model_name="hf_json_export_wordpiece",
            version=v"0.3.0",
        )

        _assert_hf_export_roundtrip(tokenizer, samples)
    end

    @testset "Unigram export/reload parity (metaspace)" begin
        tokenizer = train_unigram(
            corpus;
            vocab_size=96,
            seed_size=800,
            num_iters=2,
            max_subword_length=8,
            prune_fraction=0.2,
            special_tokens=Dict(
                :unk => "<unk>",
                :pad => "<pad>",
                :bos => "<s>",
                :eos => "</s>",
            ),
            whitespace_marker="▁",
            model_name="hf_json_export_unigram_metaspace",
            version=v"0.3.0",
        )

        _assert_hf_export_roundtrip(tokenizer, samples)
    end

    @testset "Unigram export/reload parity (no marker)" begin
        tokenizer = train_unigram(
            corpus;
            vocab_size=96,
            seed_size=800,
            num_iters=2,
            max_subword_length=8,
            prune_fraction=0.2,
            special_tokens=Dict(
                :unk => "<unk>",
                :bos => "<s>",
                :eos => "</s>",
            ),
            whitespace_marker="",
            model_name="hf_json_export_unigram_nomarker",
            version=v"0.3.0",
        )

        _assert_hf_export_roundtrip(tokenizer, ["hello", "token mix"])
    end

    @testset "SentencePiece Unigram export/reload parity" begin
        tokenizer = train_sentencepiece(
            corpus;
            vocab_size=96,
            model_type=:unigram,
            special_tokens=Dict(
                :unk => "<unk>",
                :pad => "<pad>",
                :bos => "<s>",
                :eos => "</s>",
            ),
            whitespace_marker="▁",
            seed_size=800,
            num_iters=2,
            max_subword_length=8,
            prune_fraction=0.2,
            model_name="hf_json_export_sentencepiece_unigram",
            version=v"0.3.0",
        )

        _assert_hf_export_roundtrip(tokenizer, samples)
    end

    @testset "SentencePiece BPE export/reload parity" begin
        tokenizer = train_sentencepiece(
            corpus;
            vocab_size=96,
            model_type=:bpe,
            min_frequency=1,
            special_tokens=Dict(
                :unk => "<unk>",
                :pad => "<pad>",
                :bos => "<s>",
                :eos => "</s>",
            ),
            whitespace_marker="▁",
            model_name="hf_json_export_sentencepiece_bpe",
            version=v"0.3.0",
        )

        _assert_hf_export_roundtrip(tokenizer, samples)
    end

    @testset "ByteBPE export/reload parity" begin
        byte_corpus = [
            "hello world",
            "byte level bpe works",
            "café costs 5!",
            "emoji 🙂 token",
            "punctuation, and symbols!",
        ]

        byte_samples = [
            "hello world",
            "café!",
            "emoji 🙂",
            "café 🙂 world",
        ]

        tokenizer = train_bytebpe(
            byte_corpus;
            vocab_size=384,
            min_frequency=1,
            special_tokens=Dict(
                :unk => "<UNK>",
                :pad => "<PAD>",
                :bos => "<BOS>",
                :eos => "<EOS>",
            ),
            end_of_word_marker="</w>",
            include_full_byte_alphabet=true,
            model_name="hf_json_export_bytebpe",
            version=v"0.3.0",
        )

        reloaded = _assert_hf_export_roundtrip(
            tokenizer,
            byte_samples;
            require_string_boundaries=false,
        )
        @test reloaded.explicit isa HuggingFaceJSONTokenizer
        @test reloaded.explicit.base isa ByteBPETokenizer
        @test reloaded.auto isa HuggingFaceJSONTokenizer
        @test reloaded.auto.base isa ByteBPETokenizer

        multibyte_text = "café 🙂 token"
        tokenization_text = tokenization_view(reloaded.explicit, multibyte_text)
        multibyte_result = encode_result(
            reloaded.explicit,
            tokenization_text;
            assume_normalized=true,
            add_special_tokens=true,
            return_offsets=true,
            return_masks=true,
        )
        @test multibyte_result.offsets !== nothing
        @test_nowarn assert_offsets_contract(
            tokenization_text,
            multibyte_result.offsets;
            require_string_boundaries=false,
        )
        @test offsets_are_nonoverlapping(
            multibyte_result.offsets;
            ignore_sentinel=true,
            ignore_empty=true,
        )

        nonboundary_offsets = Tuple{Int,Int}[]
        for offset in multibyte_result.offsets
            has_nonempty_span(offset) || continue
            try_span_substring(tokenization_text, offset) === nothing || continue
            push!(nonboundary_offsets, offset)
        end
        if !isempty(nonboundary_offsets)
            for offset in nonboundary_offsets
                @test_nowarn span_codeunits(tokenization_text, offset)
                @test !isempty(span_codeunits(tokenization_text, offset))
            end
        end
    end

    @testset "Existing HF tokenizer can be re-exported" begin
        tokenizer = load_tokenizer(fixture("hf_json_wordpiece", "tokenizer.json"); format=:hf_tokenizer_json)
        _assert_hf_export_roundtrip(tokenizer, ["Hello world", "Hello keemena subwords"])
    end

    @testset "TemplateProcessing special_tokens array/map compatibility" begin
        tokenizer_array = load_tokenizer(
            fixture("hf_json_wordpiece", "tokenizer.json");
            format=:hf_tokenizer_json,
        )
        tokenizer_map = load_tokenizer(
            fixture("hf_json_wordpiece_canonical", "tokenizer.json");
            format=:hf_tokenizer_json,
        )

        @test tokenizer_array isa HuggingFaceJSONTokenizer
        @test tokenizer_map isa HuggingFaceJSONTokenizer

        samples = [
            "Hello world",
            "Hello keemena subwords",
        ]

        for text in samples
            @test tokenize(tokenizer_map, text) == tokenize(tokenizer_array, text)
            for add_special_tokens in (false, true)
                ids_array = encode(tokenizer_array, text; add_special_tokens=add_special_tokens)
                ids_map = encode(tokenizer_map, text; add_special_tokens=add_special_tokens)
                @test ids_map == ids_array
                @test decode(tokenizer_map, ids_map) == decode(tokenizer_array, ids_array)
            end
        end
    end
end
