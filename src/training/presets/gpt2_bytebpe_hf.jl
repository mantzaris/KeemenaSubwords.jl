"""
    train_hf_gpt2_bytebpe(corpus; kwargs...) -> HuggingFaceJSONTokenizer

Train a GPT-2 style ByteLevel BPE tokenizer and return a
`HuggingFaceJSONTokenizer` pipeline composed of:
- No-op normalizer
- ByteLevel pre-tokenization
- ByteLevel post-processing (no BOS/EOS insertion)
- ByteLevel decoding

Special token behavior:
- By default, this preset uses a single special token:
  `special_tokens=Dict(:unk => "<|endoftext|>")`.
- `add_special_tokens=true` does not change ids by default because GPT-2 style
  ByteLevel pipelines do not auto-insert BOS/EOS.
- Special tokens present verbatim in input text can still be matched through HF
  `added_tokens` patterns.

KeemenaPreprocessing integration:
- `tokenization_text = tokenization_view(tokenizer, clean_text)`
- `encode_result(tokenizer, tokenization_text; assume_normalized=true,
  return_offsets=true, return_masks=true)`

Export/reload flow:
- `export_tokenizer(tokenizer, out_dir; format=:hf_tokenizer_json)`
- `load_hf_tokenizer_json(joinpath(out_dir, "tokenizer.json"))`
"""
function train_hf_gpt2_bytebpe(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(
        :unk => "<|endoftext|>",
    ),
    end_of_word_marker::String="</w>",
    add_prefix_space::Bool=false,
    trim_offsets::Bool=true,
    use_regex::Bool=true,
    export_unk_token_null::Bool=true,
    model_name::String="trained_hf_gpt2_bytebpe",
    version::VersionNumber=v"0.3.0",
)::HuggingFaceJSONTokenizer
    return train_hf_gpt2_bytebpe_result(
        corpus;
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        end_of_word_marker=end_of_word_marker,
        add_prefix_space=add_prefix_space,
        trim_offsets=trim_offsets,
        use_regex=use_regex,
        export_unk_token_null=export_unk_token_null,
        model_name=model_name,
        version=version,
    ).tokenizer
end

"""
    train_hf_gpt2_bytebpe_result(corpus; kwargs...) ->
        TrainingResult{HuggingFaceJSONTokenizer,GPT2ByteBPETrainingConfig,GPT2ByteBPETrainingArtifacts}

Train a GPT-2 style ByteLevel BPE tokenizer and return:
- `tokenizer::HuggingFaceJSONTokenizer`
- `config::GPT2ByteBPETrainingConfig`
- `artifacts::GPT2ByteBPETrainingArtifacts`

The returned tokenizer wraps an inner trained `ByteBPETokenizer` and preserves a
HF-native ByteLevel pipeline. By default, exported HF JSON uses
`model.unk_token = null` for GPT-2 compatibility while the internal Julia base
tokenizer still uses a concrete unknown token string.
"""
function train_hf_gpt2_bytebpe_result(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(
        :unk => "<|endoftext|>",
    ),
    end_of_word_marker::String="</w>",
    add_prefix_space::Bool=false,
    trim_offsets::Bool=true,
    use_regex::Bool=true,
    export_unk_token_null::Bool=true,
    model_name::String="trained_hf_gpt2_bytebpe",
    version::VersionNumber=v"0.3.0",
)::TrainingResult{
    HuggingFaceJSONTokenizer,
    GPT2ByteBPETrainingConfig,
    GPT2ByteBPETrainingArtifacts,
}
    config = GPT2ByteBPETrainingConfig(
        vocab_size,
        min_frequency,
        _normalize_special_tokens(special_tokens),
        String(end_of_word_marker),
        add_prefix_space,
        trim_offsets,
        use_regex,
        export_unk_token_null,
        String(model_name),
        version,
    )
    return _train_hf_gpt2_bytebpe_result_impl(corpus, config)
end

function _train_hf_gpt2_bytebpe_result_impl(
    corpus,
    config::GPT2ByteBPETrainingConfig,
)::TrainingResult{
    HuggingFaceJSONTokenizer,
    GPT2ByteBPETrainingConfig,
    GPT2ByteBPETrainingArtifacts,
}
    _validate_hf_gpt2_bytebpe_config(config)

    normalizer = HFNoopNormalizer()
    training_pretokenizer = _hf_gpt2_training_pretokenizer(config)
    inner_config = ByteBPETrainingConfig(
        config.vocab_size,
        config.min_frequency,
        config.special_tokens,
        config.end_of_word_marker,
        training_pretokenizer,
        true,
        config.model_name * "_inner",
        config.version,
    )
    inner_result = _train_bytebpe_result_impl(corpus, inner_config)
    base_tokenizer = inner_result.tokenizer

    model_unk_token = config.export_unk_token_null ? nothing : base_tokenizer.base.unk_token
    model = HFBPEModelSpec(
        copy(base_tokenizer.base.vocab.id_to_token),
        copy(inner_result.artifacts.merge_pairs),
        model_unk_token,
        true,
        nothing,
        base_tokenizer.base.end_of_word_marker,
        false,
        false,
        nothing,
    )
    pretokenizer = HFByteLevelPreTokenizer(
        config.add_prefix_space,
        config.trim_offsets,
        config.use_regex,
    )
    postprocessor = HFByteLevelPostProcessor(
        config.add_prefix_space,
        config.trim_offsets,
        config.use_regex,
    )
    decoder = HFByteLevelDecoder(
        config.add_prefix_space,
        config.trim_offsets,
        config.use_regex,
    )
    hf_added_tokens = _hf_gpt2_special_added_tokens(base_tokenizer, config.special_tokens)

    tokenizer = _build_hf_tokenizer_from_parts(
        model,
        base_tokenizer,
        normalizer,
        pretokenizer,
        postprocessor,
        decoder;
        added_tokens=hf_added_tokens,
        model_name=config.model_name,
        version=config.version,
        source_path="<trained>",
    )
    artifacts = GPT2ByteBPETrainingArtifacts(
        inner_result.artifacts,
        copy(hf_added_tokens),
    )
    return TrainingResult(tokenizer, config, artifacts)
end

function _hf_gpt2_training_pretokenizer(
    config::GPT2ByteBPETrainingConfig,
)::Function
    return text -> begin
        raw_splits, _ = _hf_bytelevel_raw_splits_with_work_spans(
            String(text);
            add_prefix_space=config.add_prefix_space,
            use_regex=config.use_regex,
        )

        out = String[]
        for split in raw_splits
            isempty(split.piece) && continue
            push!(out, split.piece)
        end
        return out
    end
end

function _hf_gpt2_special_added_tokens(
    tokenizer::ByteBPETokenizer,
    special_tokens_map::Dict{Symbol,String},
)::Vector{HFAddedToken}
    added_tokens = HFAddedToken[]
    seen_ids = Set{Int}()

    for pair in _ordered_special_token_pairs(special_tokens_map)
        token = pair.second
        token_id = token_to_id(tokenizer, token)
        token_id in seen_ids && continue
        push!(added_tokens, HFAddedToken(
            token,
            token_id,
            true,
            false,
            false,
            false,
            false,
        ))
        push!(seen_ids, token_id)
    end

    sort!(added_tokens; by=t -> (t.id, t.content))
    return added_tokens
end
