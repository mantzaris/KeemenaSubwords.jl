"""
Load tokenizer by built-in model name.
"""
function load_tokenizer(
    name::Symbol;
    prefetch::Bool=true,
    kwargs...,
)::AbstractSubwordTokenizer
    if prefetch
        prefetch_models([name])
    end
    info = describe_model(name)
    if !info.exists
        if info.distribution == :installable_gated
            throw(ArgumentError(
                "Model $name is gated and not installed. Run install_model!(:$name; token=ENV[\"HF_TOKEN\"]) first.",
            ))
        end
        throw(ArgumentError("Model asset missing on disk for $name at $(info.path)"))
    end
    return load_tokenizer(info.path; format=info.format, model_name=String(name), kwargs...)
end

"""
Load tokenizer from file system path.
"""
function load_tokenizer(
    path::AbstractString;
    format::Symbol=:auto,
    kwargs...,
)::AbstractSubwordTokenizer
    resolved = String(path)
    if format === :bpe_gpt2
        return _load_bpe_gpt2(resolved; kwargs...)
    elseif format === :tiktoken
        return load_tiktoken(resolved; kwargs...)
    elseif format === :hf_tokenizer_json
        if isdir(resolved) && !isfile(joinpath(resolved, "tokenizer.json")) && _has_gpt2_assets(resolved)
            return _load_bpe_gpt2(resolved; kwargs...)
        end
        return load_hf_tokenizer_json(resolved; kwargs...)
    end
    selected_format = format === :auto ? _detect_format(resolved) : _canonical_load_format(format)

    if selected_format === :wordpiece
        return load_wordpiece(resolved; kwargs...)
    elseif selected_format === :bpe
        return load_bpe(resolved; kwargs...)
    elseif selected_format === :bpe_gpt2
        return _load_bpe_gpt2(resolved; kwargs...)
    elseif selected_format === :bytebpe
        return load_bytebpe(resolved; kwargs...)
    elseif selected_format === :unigram
        return load_unigram(resolved; kwargs...)
    elseif selected_format === :sentencepiece
        return load_sentencepiece(resolved; kwargs...)
    elseif selected_format === :tiktoken
        return load_tiktoken(resolved; kwargs...)
    elseif selected_format === :hf_tokenizer_json
        return load_hf_tokenizer_json(resolved; kwargs...)
    end

    throw(ArgumentError("Unsupported tokenizer format: $selected_format"))
end

"""
Load tokenizer from explicit `(vocab_path, merges_path)` tuple.
"""
function load_tokenizer(
    paths::Tuple{<:AbstractString,<:AbstractString};
    format::Symbol=:bpe,
    kwargs...,
)::AbstractSubwordTokenizer
    vocab_path, merges_path = paths
    selected_format = _canonical_load_format(format)

    if selected_format === :bpe
        return load_bpe(vocab_path, merges_path; kwargs...)
    elseif selected_format === :bytebpe
        return load_bytebpe(vocab_path, merges_path; kwargs...)
    end

    throw(ArgumentError("Tuple model path loading supports only :bpe/:bpe_gpt2 and :bytebpe, got: $format"))
end

"""
Load tokenizer from a named specification.

Examples:
- `(format=:wordpiece, path="/.../vocab.txt")`
- `(format=:bpe_gpt2, vocab="/.../vocab.txt", merges="/.../merges.txt")`
"""
function load_tokenizer(spec::NamedTuple; kwargs...)::AbstractSubwordTokenizer
    haskey(spec, :format) || throw(ArgumentError("Tokenizer spec must include :format"))
    format = spec[:format]

    if haskey(spec, :path)
        return load_tokenizer(String(spec[:path]); format=format, kwargs...)
    end

    if haskey(spec, :vocab) && haskey(spec, :merges)
        return load_tokenizer((String(spec[:vocab]), String(spec[:merges])); format=format, kwargs...)
    end

    throw(ArgumentError("Tokenizer spec requires either :path or (:vocab, :merges)"))
end

"""
Save tokenizer to a canonical on-disk format.

`format=:internal` chooses a tokenizer-family specific default:
- `WordPieceTokenizer` -> `vocab.txt`
- `BPETokenizer` / `ByteBPETokenizer` -> `vocab.txt` + `merges.txt`
- `UnigramTokenizer` -> `unigram.tsv`
- `SentencePieceTokenizer` -> `spm.model`
"""
function save_tokenizer(
    tokenizer::AbstractSubwordTokenizer,
    outdir::AbstractString;
    format::Symbol=:internal,
)::Nothing
    export_tokenizer(tokenizer, outdir; format=format)
    return nothing
end

"""
Export tokenizer to external formats.

Supported `format` values:
- `:internal`
- `:bpe` / `:bpe_gpt2`
- `:wordpiece_vocab`
- `:unigram_tsv`
- `:sentencepiece_model`
"""
function export_tokenizer(
    tokenizer::AbstractSubwordTokenizer,
    outdir::AbstractString;
    format::Symbol,
)::Nothing
    mkpath(outdir)
    target = _canonical_export_format(tokenizer, format)

    if target === :wordpiece_vocab
        _write_wordpiece_vocab(_as_wordpiece(tokenizer), joinpath(outdir, "vocab.txt"))
    elseif target === :bpe_files
        _write_bpe_files(_as_bpe(tokenizer), outdir)
    elseif target === :unigram_tsv
        _write_unigram_tsv(_as_unigram(tokenizer), joinpath(outdir, "unigram.tsv"))
    elseif target === :sentencepiece_model
        _write_sentencepiece_model(tokenizer, joinpath(outdir, "spm.model"))
    else
        throw(ArgumentError("Unsupported export target: $target"))
    end

    return nothing
end

function _canonical_load_format(format::Symbol)::Symbol
    if format in (:bpe, :bpe_gpt2)
        return :bpe
    elseif format in (:bytebpe,)
        return :bytebpe
    elseif format in (:wordpiece, :wordpiece_vocab)
        return :wordpiece
    elseif format in (:unigram, :unigram_tsv)
        return :unigram
    elseif format in (:sentencepiece, :sentencepiece_model)
        return :sentencepiece
    elseif format in (:tiktoken,)
        return :tiktoken
    elseif format in (:hf_tokenizer_json,)
        return :hf_tokenizer_json
    end

    throw(ArgumentError("Unsupported tokenizer format: $format"))
end

function _canonical_export_format(tokenizer::AbstractSubwordTokenizer, format::Symbol)::Symbol
    if format === :internal
        if tokenizer isa WordPieceTokenizer
            return :wordpiece_vocab
        elseif tokenizer isa BPETokenizer || tokenizer isa ByteBPETokenizer
            return :bpe_files
        elseif tokenizer isa UnigramTokenizer
            return :unigram_tsv
        elseif tokenizer isa SentencePieceTokenizer
            return :sentencepiece_model
        end
        throw(ArgumentError("No internal save format is defined for $(typeof(tokenizer))"))
    end

    if format in (:bpe, :bpe_gpt2)
        return :bpe_files
    elseif format in (:wordpiece, :wordpiece_vocab)
        return :wordpiece_vocab
    elseif format in (:unigram, :unigram_tsv)
        return :unigram_tsv
    elseif format in (:sentencepiece, :sentencepiece_model)
        return :sentencepiece_model
    end

    throw(ArgumentError("Unsupported export format: $format"))
end

function _detect_format(path::String)::Symbol
    if isdir(path)
        has_vocab = isfile(joinpath(path, "vocab.txt"))
        has_merges = isfile(joinpath(path, "merges.txt"))
        has_vocab_json = isfile(joinpath(path, "vocab.json"))
        has_encoder_json = isfile(joinpath(path, "encoder.json"))
        has_vocab_bpe = isfile(joinpath(path, "vocab.bpe"))
        has_tokenizer_json = isfile(joinpath(path, "tokenizer.json"))
        has_sentencepiece_model = any(isfile, [
            joinpath(path, "spm.model"),
            joinpath(path, "tokenizer.model"),
            joinpath(path, "tokenizer.model.v3"),
            joinpath(path, "sentencepiece.bpe.model"),
        ])
        tiktoken_files = filter(p -> endswith(lowercase(p), ".tiktoken"), readdir(path; join=true))

        if has_tokenizer_json
            return :hf_tokenizer_json
        elseif (has_vocab_json && has_merges) || (has_encoder_json && has_vocab_bpe)
            return :bpe_gpt2
        elseif has_vocab && has_merges
            return :bpe
        elseif has_vocab
            return :wordpiece
        elseif isfile(joinpath(path, "unigram.tsv"))
            return :unigram
        elseif has_sentencepiece_model
            return :sentencepiece
        elseif length(tiktoken_files) == 1
            return :tiktoken
        end

        throw(ArgumentError("Could not infer tokenizer format from directory: $path"))
    end

    if !isfile(path)
        throw(ArgumentError("Tokenizer path does not exist: $path"))
    end

    lower_path = lowercase(path)
    lower_name = lowercase(basename(path))

    if endswith(lower_path, ".model") || endswith(lower_path, ".model.v3")
        return :sentencepiece
    elseif endswith(lower_path, ".tiktoken")
        return :tiktoken
    elseif lower_name == "tokenizer.json"
        return :hf_tokenizer_json
    elseif lower_name == "vocab.json"
        sibling_merges = joinpath(dirname(path), "merges.txt")
        isfile(sibling_merges) || throw(ArgumentError("Found vocab.json without sibling merges.txt: $path"))
        return :bpe_gpt2
    elseif lower_name == "encoder.json"
        sibling_merges = joinpath(dirname(path), "vocab.bpe")
        isfile(sibling_merges) || throw(ArgumentError("Found encoder.json without sibling vocab.bpe: $path"))
        return :bpe_gpt2
    elseif lower_name == "vocab.bpe"
        sibling_vocab = joinpath(dirname(path), "encoder.json")
        isfile(sibling_vocab) || throw(ArgumentError("Found vocab.bpe without sibling encoder.json: $path"))
        return :bpe_gpt2
    elseif lower_name == "vocab.txt"
        sibling_merges = joinpath(dirname(path), "merges.txt")
        return isfile(sibling_merges) ? :bpe : :wordpiece
    elseif lower_name == "unigram.tsv"
        return :unigram
    end

    throw(ArgumentError("Could not infer tokenizer format from file: $path"))
end

function _has_gpt2_assets(path::AbstractString)::Bool
    return (
        (isfile(joinpath(path, "vocab.json")) && isfile(joinpath(path, "merges.txt"))) ||
        (isfile(joinpath(path, "encoder.json")) && isfile(joinpath(path, "vocab.bpe")))
    )
end

function _as_wordpiece(tokenizer::AbstractSubwordTokenizer)::WordPieceTokenizer
    tokenizer isa WordPieceTokenizer || throw(ArgumentError("Format :wordpiece_vocab requires WordPieceTokenizer, got $(typeof(tokenizer))"))
    return tokenizer
end

function _as_bpe(tokenizer::AbstractSubwordTokenizer)::BPETokenizer
    if tokenizer isa BPETokenizer
        return tokenizer
    elseif tokenizer isa ByteBPETokenizer
        return tokenizer.base
    elseif tokenizer isa SentencePieceTokenizer && tokenizer.inner isa BPETokenizer
        return tokenizer.inner
    end

    throw(ArgumentError("Requested BPE export from incompatible tokenizer: $(typeof(tokenizer))"))
end

function _as_unigram(tokenizer::AbstractSubwordTokenizer)::UnigramTokenizer
    if tokenizer isa UnigramTokenizer
        return tokenizer
    elseif tokenizer isa SentencePieceTokenizer && tokenizer.inner isa UnigramTokenizer
        return tokenizer.inner
    end

    throw(ArgumentError("Requested Unigram export from incompatible tokenizer: $(typeof(tokenizer))"))
end

function _write_wordpiece_vocab(tokenizer::WordPieceTokenizer, path::String)::Nothing
    _write_token_lines(path, tokenizer.vocab.id_to_token)
    return nothing
end

function _write_bpe_files(tokenizer::BPETokenizer, outdir::AbstractString)::Nothing
    vocab_path = joinpath(outdir, "vocab.txt")
    merges_path = joinpath(outdir, "merges.txt")

    _write_token_lines(vocab_path, tokenizer.vocab.id_to_token)

    pairs = _sorted_merge_pairs(tokenizer)
    open(merges_path, "w") do io
        println(io, "# merges")
        for (left, right) in pairs
            println(io, left, " ", right)
        end
    end

    return nothing
end

function _write_unigram_tsv(tokenizer::UnigramTokenizer, path::String)::Nothing
    open(path, "w") do io
        for id in 1:length(tokenizer.vocab.id_to_token)
            token = tokenizer.vocab.id_to_token[id]
            score = tokenizer.logprobs[id]
            special = _special_symbol_for_id(tokenizer.vocab, id)
            if special === nothing
                println(io, token, "\t", score)
            else
                println(io, token, "\t", score, "\t", special)
            end
        end
    end

    return nothing
end

function _write_sentencepiece_model(tokenizer::AbstractSubwordTokenizer, path::String)::Nothing
    if tokenizer isa SentencePieceTokenizer
        _write_sentencepiece_model_from_parts(tokenizer.inner, tokenizer.whitespace_marker, path)
        return nothing
    elseif tokenizer isa UnigramTokenizer
        marker = isempty(tokenizer.whitespace_marker) ? "▁" : tokenizer.whitespace_marker
        _write_sentencepiece_model_from_parts(tokenizer, marker, path)
        return nothing
    elseif tokenizer isa BPETokenizer
        _write_sentencepiece_model_from_parts(tokenizer, "▁", path)
        return nothing
    elseif tokenizer isa ByteBPETokenizer
        _write_sentencepiece_model_from_parts(tokenizer.base, "▁", path)
        return nothing
    end

    throw(ArgumentError("Cannot export $(typeof(tokenizer)) as :sentencepiece_model"))
end

function _write_sentencepiece_model_from_parts(inner::UnigramTokenizer, marker::String, path::String)::Nothing
    open(path, "w") do io
        println(io, "type=unigram")
        println(io, "whitespace_marker=", marker)
        println(io, "unk_token=", inner.unk_token)

        for id in 1:length(inner.vocab.id_to_token)
            token = inner.vocab.id_to_token[id]
            score = inner.logprobs[id]
            special = _special_symbol_for_id(inner.vocab, id)
            if special === nothing
                println(io, "piece\t", token, "\t", score)
            else
                println(io, "piece\t", token, "\t", score, "\t", special)
            end
        end
    end

    return nothing
end

function _write_sentencepiece_model_from_parts(inner::BPETokenizer, marker::String, path::String)::Nothing
    open(path, "w") do io
        println(io, "type=bpe")
        println(io, "whitespace_marker=", marker)
        println(io, "unk_token=", inner.unk_token)

        for id in 1:length(inner.vocab.id_to_token)
            token = inner.vocab.id_to_token[id]
            special = _special_symbol_for_id(inner.vocab, id)
            if special === nothing
                println(io, "piece\t", token, "\t", -1.0)
            else
                println(io, "piece\t", token, "\t", -1.0, "\t", special)
            end
        end

        for (left, right) in _sorted_merge_pairs(inner)
            println(io, "merge\t", left, "\t", right)
        end
    end

    return nothing
end

function _sorted_merge_pairs(tokenizer::BPETokenizer)::Vector{Tuple{String,String}}
    ranked = collect(tokenizer.pair_ranks)
    sort!(ranked; by=last)
    return [pair for (pair, _) in ranked]
end

function _special_symbol_for_id(vocab::SubwordVocabulary, id::Int)::Union{Nothing,Symbol}
    for (symbol, sid) in vocab.special_token_ids
        sid == id && return symbol
    end
    return nothing
end

function _write_token_lines(path::String, tokens::Vector{String})::Nothing
    open(path, "w") do io
        for token in tokens
            println(io, token)
        end
    end
    return nothing
end

function _load_bpe_gpt2(
    path::AbstractString;
    unk_token::AbstractString="<unk>",
    model_name::Union{Nothing,AbstractString}=nothing,
    kwargs...,
)::ByteBPETokenizer
    _ = kwargs
    vocab_json_path, merges_path = _resolve_gpt2_paths(path)
    tokens = _read_gpt2_vocab_json(vocab_json_path)

    if !(String(unk_token) in tokens)
        pushfirst!(tokens, String(unk_token))
    end

    special_map = detect_special_tokens(tokens, String(unk_token))
    vocab = build_vocab(tokens; special_tokens=special_map)
    pairs = _read_merge_pairs(merges_path)
    pair_ranks = Dict{Tuple{String,String},Int}()
    for (i, pair) in enumerate(pairs)
        pair_ranks[pair] = i
    end

    name = model_name === nothing ? basename(vocab_json_path) : String(model_name)
    base_metadata = TokenizerMetadata(:bpe_gpt2, name, v"0.2.0", :none)
    base = BPETokenizer(vocab, pair_ranks, String(unk_token), nothing, base_metadata)

    b2u, u2b = _byte_unicode_tables()
    metadata = TokenizerMetadata(:bytebpe, name, v"0.2.0", :none)
    return ByteBPETokenizer(base, b2u, u2b, metadata)
end

function _resolve_gpt2_paths(path::AbstractString)::Tuple{String,String}
    if isdir(path)
        vocab = joinpath(path, "vocab.json")
        merges = joinpath(path, "merges.txt")
        encoder = joinpath(path, "encoder.json")
        vocab_bpe = joinpath(path, "vocab.bpe")

        if isfile(vocab) && isfile(merges)
            return (vocab, merges)
        elseif isfile(encoder) && isfile(vocab_bpe)
            return (encoder, vocab_bpe)
        end

        throw(ArgumentError("Missing GPT2 BPE files in directory: $path (expected vocab.json+merges.txt or encoder.json+vocab.bpe)"))
    end

    isfile(path) || throw(ArgumentError("GPT2 BPE path does not exist: $path"))
    lower_name = lowercase(basename(path))

    if lower_name == "vocab.json"
        merges = joinpath(dirname(path), "merges.txt")
        isfile(merges) || throw(ArgumentError("Missing merges.txt next to $path"))
        return (String(path), merges)
    elseif lower_name == "encoder.json"
        merges = joinpath(dirname(path), "vocab.bpe")
        isfile(merges) || throw(ArgumentError("Missing vocab.bpe next to $path"))
        return (String(path), merges)
    elseif lower_name == "vocab.bpe"
        vocab = joinpath(dirname(path), "encoder.json")
        isfile(vocab) || throw(ArgumentError("Missing encoder.json next to $path"))
        return (vocab, String(path))
    end

    throw(ArgumentError("GPT2 BPE file path must point to vocab.json, encoder.json, vocab.bpe, or a model directory: $path"))
end

function _read_gpt2_vocab_json(path::AbstractString)::Vector{String}
    pair_re = r"\"((?:\\.|[^\"])*)\"\s*:\s*(\d+)"
    pairs = Tuple{String,Int}[]

    content = read(path, String)
    for m in eachmatch(pair_re, content)
        token_raw = m.captures[1]
        id = parse(Int, m.captures[2])
        token = Base.unescape_string(token_raw)
        push!(pairs, (token, id))
    end

    isempty(pairs) && throw(ArgumentError("No token->id entries found in vocab.json: $path"))
    max_id = maximum(last(pair) for pair in pairs)
    tokens = Vector{Union{Nothing,String}}(undef, max_id + 1)
    fill!(tokens, nothing)

    for (token, id) in pairs
        idx = id + 1
        (1 <= idx <= length(tokens)) || throw(ArgumentError("Out-of-range token id $id in $path"))
        tokens[idx] === nothing || throw(ArgumentError("Duplicate token id $id in $path"))
        tokens[idx] = token
    end

    any(t -> t === nothing, tokens) && throw(ArgumentError("vocab.json has missing token ids in 0:$(max_id): $path"))
    return String[t::String for t in tokens]
end
