"""
SentencePiece training stub.

The public entrypoint exists for forward-compatibility but training is not
implemented yet.
"""
function _train_sentencepiece_impl(
    corpus;
    vocab_size::Int,
    model_type::Symbol=:unigram,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<unk>", :pad => "<pad>"),
)::SentencePieceTokenizer
    _ = corpus
    _ = vocab_size
    _ = model_type
    _ = special_tokens
    throw(ArgumentError("train_sentencepiece is not implemented yet. Use load_tokenizer(...) with an existing model for now."))
end
