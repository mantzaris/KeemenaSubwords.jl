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

    return nothing
end

function _assert_hf_export_roundtrip(
    tokenizer::AbstractSubwordTokenizer,
    samples::Vector{String};
    add_special_modes::Tuple{Vararg{Bool}}=(false, true),
)::Nothing
    outdir = mktempdir()
    export_tokenizer(tokenizer, outdir; format=:hf_tokenizer_json)

    tokenizer_json_path = joinpath(outdir, "tokenizer.json")
    @test isfile(tokenizer_json_path)
    _assert_exported_hf_json_sanity(tokenizer_json_path)

    reloaded = load_tokenizer(outdir; format=:hf_tokenizer_json)
    @test reloaded isa HuggingFaceJSONTokenizer

    for text in samples
        @test tokenize(reloaded, text) == tokenize(tokenizer, text)

        for add_special_tokens in add_special_modes
            ids_original = encode(tokenizer, text; add_special_tokens=add_special_tokens)
            ids_reloaded = encode(reloaded, text; add_special_tokens=add_special_tokens)
            @test ids_reloaded == ids_original
            @test decode(reloaded, ids_reloaded) == decode(tokenizer, ids_original)

            tokenization_text = tokenization_view(reloaded, text)
            result = encode_result(
                reloaded,
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
                require_string_boundaries=true,
            )
            @test offsets_are_nonoverlapping(
                result.offsets;
                ignore_sentinel=true,
                ignore_empty=true,
            )
        end
    end

    return nothing
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

    @testset "Existing HF tokenizer can be re-exported" begin
        tokenizer = load_tokenizer(fixture("hf_json_wordpiece", "tokenizer.json"); format=:hf_tokenizer_json)
        _assert_hf_export_roundtrip(tokenizer, ["Hello world", "Hello keemena subwords"])
    end
end
