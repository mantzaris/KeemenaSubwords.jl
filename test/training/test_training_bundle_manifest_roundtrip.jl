using JSON3

function _bundle_json_has(obj, key::String)::Bool
    return haskey(obj, key) || haskey(obj, Symbol(key))
end

function _bundle_json_get(obj, key::String)
    if haskey(obj, key)
        return obj[key]
    elseif haskey(obj, Symbol(key))
        return obj[Symbol(key)]
    end
    throw(ArgumentError("Missing key '$key' in JSON object"))
end

function _assert_bundle_manifest_shape(
    bundle_dir::String;
    expected_load_format::Symbol,
    expected_file_keys::Vector{String},
    expected_load_kw_keys::Vector{String},
)::Nothing
    manifest_path = joinpath(bundle_dir, "keemena_training_manifest.json")
    @test isfile(manifest_path)

    root = JSON3.read(read(manifest_path, String))
    @test Int(_bundle_json_get(root, "schema_version")) == 1
    @test Symbol(String(_bundle_json_get(root, "load_format"))) == expected_load_format

    for key in ("trainer", "export_format", "files", "load_kwargs", "training_config", "metadata", "warnings")
        @test _bundle_json_has(root, key)
    end

    files = _bundle_json_get(root, "files")
    for key in expected_file_keys
        @test _bundle_json_has(files, key)
        relpath = String(_bundle_json_get(files, key))
        @test isfile(joinpath(bundle_dir, relpath))
    end

    load_kwargs = _bundle_json_get(root, "load_kwargs")
    for key in expected_load_kw_keys
        @test _bundle_json_has(load_kwargs, key)
    end
end

function _assert_bundle_roundtrip_parity(
    original,
    reloaded,
    samples::Vector{String};
    add_special_modes::Tuple{Vararg{Bool}}=(false, true),
    require_string_boundaries::Bool=true,
)::Nothing
    @test typeof(reloaded) == typeof(original)

    for text in samples
        @test tokenize(reloaded, text) == tokenize(original, text)

        for add_special_tokens in add_special_modes
            ids_original = encode(original, text; add_special_tokens=add_special_tokens)
            ids_reloaded = encode(reloaded, text; add_special_tokens=add_special_tokens)
            @test ids_reloaded == ids_original
            @test decode(reloaded, ids_reloaded) == decode(original, ids_original)

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
                require_string_boundaries=require_string_boundaries,
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

@testset "Training bundle manifest roundtrip" begin
    base_corpus = [
        "hello world",
        "hello, world!",
        "café costs 5 euros",
        "emoji 🙂 token",
        "你好 世界",
        "hello world hello world",
    ]

    cases = [
        (
            name="BPE",
            train=() -> train_bpe_result(
                base_corpus;
                vocab_size=128,
                min_frequency=1,
                special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>", :bos => "<BOS>", :eos => "<EOS>"),
                end_of_word_marker="</w>",
                model_name="bundle_bpe",
                version=v"0.3.0",
            ),
            samples=["hello world", "hello, world!", "café costs"],
            expected_load_format=:bpe,
            expected_file_keys=["vocab_txt", "merges_txt"],
            expected_load_kw_keys=["unk_token", "end_of_word_marker", "special_tokens"],
            require_string_boundaries=true,
            add_special_modes=(false, true),
        ),
        (
            name="ByteBPE",
            train=() -> train_bytebpe_result(
                base_corpus;
                vocab_size=320,
                min_frequency=1,
                special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>", :bos => "<BOS>", :eos => "<EOS>"),
                end_of_word_marker="</w>",
                include_full_byte_alphabet=true,
                model_name="bundle_bytebpe",
                version=v"0.3.0",
            ),
            samples=["hello world", "café 🙂", "emoji 🙂 token"],
            expected_load_format=:bytebpe,
            expected_file_keys=["vocab_txt", "merges_txt"],
            expected_load_kw_keys=["unk_token", "end_of_word_marker", "special_tokens"],
            require_string_boundaries=false,
            add_special_modes=(false, true),
        ),
        (
            name="WordPiece",
            train=() -> train_wordpiece_result(
                base_corpus;
                vocab_size=160,
                min_frequency=1,
                special_tokens=Dict(
                    :unk => "[UNK]",
                    :pad => "[PAD]",
                    :cls => "[CLS]",
                    :sep => "[SEP]",
                    :mask => "[MASK]",
                ),
                continuation_prefix="##",
                max_input_chars_per_word=100,
                model_name="bundle_wordpiece",
                version=v"0.3.0",
            ),
            samples=["hello world", "hello café", "hello, world!"],
            expected_load_format=:wordpiece,
            expected_file_keys=["vocab_txt"],
            expected_load_kw_keys=["continuation_prefix", "unk_token", "max_input_chars_per_word", "special_tokens"],
            require_string_boundaries=true,
            add_special_modes=(false, true),
        ),
        (
            name="Unigram",
            train=() -> train_unigram_result(
                base_corpus;
                vocab_size=128,
                seed_size=600,
                num_iters=2,
                max_subword_length=8,
                prune_fraction=0.2,
                special_tokens=Dict(:unk => "<unk>", :pad => "<pad>", :bos => "<s>", :eos => "</s>"),
                whitespace_marker="▁",
                model_name="bundle_unigram",
                version=v"0.3.0",
            ),
            samples=["hello world", "café costs", "hello, world!"],
            expected_load_format=:unigram,
            expected_file_keys=["unigram_tsv"],
            expected_load_kw_keys=["unk_token", "whitespace_marker", "special_tokens"],
            require_string_boundaries=true,
            add_special_modes=(false, true),
        ),
        (
            name="SentencePiece",
            train=() -> train_sentencepiece_result(
                base_corpus;
                vocab_size=128,
                model_type=:unigram,
                special_tokens=Dict(:unk => "<unk>", :pad => "<pad>", :bos => "<s>", :eos => "</s>"),
                whitespace_marker="▁",
                seed_size=600,
                num_iters=2,
                max_subword_length=8,
                prune_fraction=0.2,
                model_name="bundle_sentencepiece",
                version=v"0.3.0",
            ),
            samples=["hello world", "café costs", "emoji token"],
            expected_load_format=:sentencepiece_model,
            expected_file_keys=["spm_model"],
            expected_load_kw_keys=["special_tokens"],
            require_string_boundaries=true,
            add_special_modes=(false, true),
        ),
        (
            name="HF BERT WordPiece",
            train=() -> train_hf_bert_wordpiece_result(
                base_corpus;
                vocab_size=160,
                min_frequency=1,
                lowercase=true,
                strip_accents=nothing,
                handle_chinese_chars=true,
                clean_text=true,
                model_name="bundle_hf_bert_wordpiece",
                version=v"0.3.0",
            ),
            samples=["Hello world", "Café naïve", "Hello,world!"],
            expected_load_format=:hf_tokenizer_json,
            expected_file_keys=["tokenizer_json"],
            expected_load_kw_keys=String[],
            require_string_boundaries=true,
            add_special_modes=(false, true),
        ),
        (
            name="HF RoBERTa ByteBPE",
            train=() -> train_hf_roberta_bytebpe_result(
                base_corpus;
                vocab_size=384,
                min_frequency=1,
                special_tokens=Dict(
                    :unk => "<unk>",
                    :pad => "<pad>",
                    :bos => "<s>",
                    :eos => "</s>",
                    :mask => "<mask>",
                ),
                add_prefix_space=true,
                trim_offsets=true,
                use_regex=true,
                model_name="bundle_hf_roberta_bytebpe",
                version=v"0.3.0",
            ),
            samples=["Hello my friend, how is your day going?", "Hello there      dear", "café 🙂"],
            expected_load_format=:hf_tokenizer_json,
            expected_file_keys=["tokenizer_json"],
            expected_load_kw_keys=String[],
            require_string_boundaries=false,
            add_special_modes=(false, true),
        ),
        (
            name="HF GPT-2 ByteBPE",
            train=() -> train_hf_gpt2_bytebpe_result(
                base_corpus;
                vocab_size=384,
                min_frequency=1,
                special_tokens=Dict(:unk => "<|endoftext|>"),
                add_prefix_space=false,
                trim_offsets=true,
                use_regex=true,
                export_unk_token_null=true,
                model_name="bundle_hf_gpt2_bytebpe",
                version=v"0.3.0",
            ),
            samples=["Hello my friend, how is your day going?", "Hello there\nHello there", "start <|endoftext|> end"],
            expected_load_format=:hf_tokenizer_json,
            expected_file_keys=["tokenizer_json"],
            expected_load_kw_keys=String[],
            require_string_boundaries=false,
            add_special_modes=(false, true),
        ),
    ]

    for case in cases
        @testset "$(case.name)" begin
            result = case.train()
            outroot = mktempdir()
            bundle_dir = joinpath(outroot, "bundle")

            save_training_bundle(result, bundle_dir)
            manifest = read_training_manifest(bundle_dir)
            @test manifest.schema_version == 1

            _assert_bundle_manifest_shape(
                bundle_dir;
                expected_load_format=case.expected_load_format,
                expected_file_keys=case.expected_file_keys,
                expected_load_kw_keys=case.expected_load_kw_keys,
            )

            reloaded = load_training_bundle(bundle_dir)
            _assert_bundle_roundtrip_parity(
                result.tokenizer,
                reloaded,
                case.samples;
                add_special_modes=case.add_special_modes,
                require_string_boundaries=case.require_string_boundaries,
            )
        end
    end
end

@testset "Training bundle manifest pretokenizer warning" begin
    corpus = ["hello world", "hello there"]
    custom_pretokenizer = text -> split(lowercase(String(text)))

    result = train_bpe_result(
        corpus;
        vocab_size=64,
        min_frequency=1,
        special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>"),
        pretokenizer=custom_pretokenizer,
        model_name="bundle_pretok_warning",
        version=v"0.3.0",
    )

    outroot = mktempdir()
    bundle_dir = joinpath(outroot, "bundle")
    save_training_bundle(result, bundle_dir)

    manifest = read_training_manifest(bundle_dir)
    @test !isempty(manifest.warnings)
    @test any(occursin("pretokenizer", warning) for warning in manifest.warnings)
    @test !haskey(manifest.training_config, "pretokenizer")
end
