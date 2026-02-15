function _map_bytelevel_piece_local(text::String)::String
    byte_to_unicode, _ = KeemenaSubwords._byte_unicode_tables()
    out = IOBuffer()
    for byte in codeunits(text)
        write(out, byte_to_unicode[Int(byte) + 1])
    end
    return String(take!(out))
end

function _assert_bytelevel_regex_split_case(
    text::String,
    expected_raw_splits::Vector{String},
)::Nothing
    splits = KeemenaSubwords._hf_bytelevel_pretokenize_with_spans(
        text;
        add_prefix_space=false,
        use_regex=true,
    )

    @test length(splits) == length(expected_raw_splits)
    spans = Tuple{Int,Int}[span for (_, span) in splits]
    @test_nowarn assert_offsets_contract(
        text,
        spans;
        require_string_boundaries=true,
    )
    @test offsets_are_nonoverlapping(
        spans;
        ignore_sentinel=true,
        ignore_empty=true,
    )

    for i in eachindex(splits)
        mapped_piece, span = splits[i]
        raw_piece = try_span_substring(text, span)
        @test raw_piece !== nothing
        @test raw_piece == expected_raw_splits[i]
        @test mapped_piece == _map_bytelevel_piece_local(raw_piece)
    end

    return nothing
end

@testset "Section 26b HF ByteLevel GPT-2 regex splitting" begin
    _assert_bytelevel_regex_split_case(
        "Hello my friend, how is your day going?",
        ["Hello", " my", " friend", ",", " how", " is", " your", " day", " going", "?"],
    )

    _assert_bytelevel_regex_split_case(
        "Hello there\nHello there",
        ["Hello", " there", "\n", "Hello", " there"],
    )

    _assert_bytelevel_regex_split_case(
        "Hello there      dear",
        ["Hello", " there", "     ", " dear"],
    )

    prefixed = KeemenaSubwords._hf_bytelevel_pretokenize_with_spans(
        "Hello";
        add_prefix_space=true,
        use_regex=true,
    )
    @test length(prefixed) == 1
    mapped_piece, span = prefixed[1]
    @test span == (1, ncodeunits("Hello") + 1)
    @test mapped_piece == _map_bytelevel_piece_local(" Hello")
end
