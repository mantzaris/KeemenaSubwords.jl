# Offsets Alignment Examples

This page is a practical tutorial for applying the normalization and offsets contract.
For the normative specification, see [Normalization and Offsets Contract](normalization_offsets_contract.md).

## Mental Model And Coordinate System

Offsets are always interpreted in the coordinate system of `tokenization_text`.
`tokenization_text` may differ from `clean_text` when tokenizer intrinsic normalization is active.

Safe pattern with KeemenaPreprocessing output:

1. `tokenization_text = tokenization_view(tokenizer, clean_text)`
2. `encode_result(tokenizer, tokenization_text; assume_normalized=true, return_offsets=true, ...)`

Offset contract reminders:

- Offsets are 1-based UTF-8 codeunit half-open spans `(start, stop)`.
- `stop` is exclusive.
- Sentinel `(0, 0)` means "no span" and should be treated as non-aligning.

## Example 1: Inspect `encode_result` Output

```@example offsets_alignment
using KeemenaSubwords

tokenizer = load_tokenizer(:core_sentencepiece_unigram_en)
clean_text = "Hello, world! This is an offsets demo."
tokenization_text = tokenization_view(tokenizer, clean_text)

result = encode_result(
    tokenizer,
    tokenization_text;
    assume_normalized = true,
    add_special_tokens = true,
    return_offsets = true,
    return_masks = true,
)

@assert result.offsets !== nothing
@assert result.special_tokens_mask !== nothing

token_offsets = result.offsets
special_tokens_mask = result.special_tokens_mask

rows = [
    (
        token_index = i,
        token_id = result.ids[i],
        token_string = result.tokens[i],
        offset = token_offsets[i],
        substring_or_nothing = try_span_substring(tokenization_text, token_offsets[i]),
        is_special = special_tokens_mask[i] == 1,
        has_span = has_nonempty_span(token_offsets[i]),
    )
    for i in eachindex(result.ids)
]

rows[1:min(end, 30)]
```

How to interpret these rows:

- Tokens with offset `(0, 0)` are no-span tokens. They are usually inserted specials.
- `is_special` and `has_span` are related but not identical concepts. Align by span, not by mask alone.
- `substring_or_nothing` helps verify offsets quickly against `tokenization_text`.
- Use `tokenization_text` for offset-based slicing. Do not assume `clean_text` uses the same coordinates.

## Example 2: Word Offsets And Subword-To-Word Mapping

```@example offsets_alignment
function whitespace_word_offsets(text)::Vector{Tuple{Int,Int}}
    offsets = Tuple{Int,Int}[]
    stop_exclusive = ncodeunits(text) + 1
    i = firstindex(text)

    while i < stop_exclusive
        while i < stop_exclusive && isspace(text[i])
            i = nextind(text, i)
        end
        i < stop_exclusive || break

        word_start = i
        while i < stop_exclusive && !isspace(text[i])
            i = nextind(text, i)
        end
        word_stop = i
        push!(offsets, (word_start, word_stop))
    end

    return offsets
end

overlap_len(a_start, a_stop, b_start, b_stop)::Int =
    max(0, min(a_stop, b_stop) - max(a_start, b_start))

function subword_to_word_index(
    word_offsets::Vector{Tuple{Int,Int}},
    subword_offset::Tuple{Int,Int},
)::Union{Nothing,Int}
    has_nonempty_span(subword_offset) || return nothing
    sub_start, sub_stop = subword_offset

    for (word_index, (word_start, word_stop)) in pairs(word_offsets)
        if sub_start >= word_start && sub_stop <= word_stop
            return word_index
        end
    end

    best_index = nothing
    best_overlap = 0
    for (word_index, (word_start, word_stop)) in pairs(word_offsets)
        current_overlap = overlap_len(sub_start, sub_stop, word_start, word_stop)
        # Strict > means equal-overlap ties keep the earliest word index.
        if current_overlap > best_overlap
            best_overlap = current_overlap
            best_index = word_index
        end
    end

    return best_overlap > 0 ? best_index : nothing
end

word_offsets = whitespace_word_offsets(tokenization_text)
token_to_word = map(token_offsets) do off
    subword_to_word_index(word_offsets, off)
end

word_rows = [
    (
        word_index = i,
        offset = word_offsets[i],
        word_substring = try_span_substring(tokenization_text, word_offsets[i]),
    )
    for i in eachindex(word_offsets)
]

token_rows = [
    (
        token_index = i,
        token_string = result.tokens[i],
        offset = token_offsets[i],
        word_index = token_to_word[i],
        word_substring = token_to_word[i] === nothing ? nothing :
            try_span_substring(tokenization_text, word_offsets[token_to_word[i]]),
    )
    for i in eachindex(result.tokens)
]

(
    words = word_rows,
    first_tokens = token_rows[1:min(end, 30)],
)
```

Limitations of this tutorial mapping:

- Subword spans can overlap multiple words in some normalization and punctuation situations.
- This example returns one best word index (full containment first, else maximum overlap).
- Equal-overlap ties are resolved to the earliest word index.
- If you need multi-word mapping, return all overlapping word indices instead of one index.

## Example 3: Special Tokens Policy For Alignment

```@example offsets_alignment
participates_in_alignment(offset, is_special)::Bool = has_nonempty_span(offset)

alignment_rows = [
    (
        token_index = i,
        token_string = result.tokens[i],
        offset = token_offsets[i],
        is_special = special_tokens_mask[i] == 1,
        participates = participates_in_alignment(token_offsets[i], special_tokens_mask[i] == 1),
    )
    for i in eachindex(result.tokens)
]

(
    skipped = [row for row in alignment_rows if !row.participates][1:min(end, 10)],
    participating = [row for row in alignment_rows if row.participates][1:min(end, 10)],
)
```

Policy summary:

- Pragmatic default: participate in alignment iff `has_nonempty_span(offset)`.
- Inserted special tokens usually have `(0, 0)` and are skipped automatically.

## Example 4: Byte-Level Caveat And Safe Extraction

Byte-level tokenizers can produce offsets that are valid codeunit spans but are not always safe Julia string slicing boundaries on multibyte text.

When you consume offsets, use this safe pattern:

```julia
# non-executable byte-level pattern
substring = try_span_substring(tokenization_text, offset)

if substring === nothing && has_nonempty_span(offset)
    bytes = span_codeunits(tokenization_text, offset)
    # Use bytes in a byte-aware path when boundaries are not string-safe.
end
```

This fallback keeps alignment pipelines robust across both string-safe and byte-level offset cases.
