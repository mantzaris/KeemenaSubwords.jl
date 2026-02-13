"""
WordPiece training stub.

The public entrypoint exists for forward-compatibility but training is not
implemented yet.
"""
function _train_wordpiece_impl(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "[UNK]", :pad => "[PAD]"),
    continuation_prefix::String="##",
)::WordPieceTokenizer
    _ = corpus
    _ = vocab_size
    _ = min_frequency
    _ = special_tokens
    _ = continuation_prefix
    throw(ArgumentError("train_wordpiece is not implemented yet. Use load_tokenizer(...) with an existing model for now."))
end
