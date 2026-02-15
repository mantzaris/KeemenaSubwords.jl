function _map_bytelevel_piece_no_regex_local(text::String)::String
    byte_to_unicode, _ = KeemenaSubwords._byte_unicode_tables()
    out = IOBuffer()
    for byte in codeunits(text)
        write(out, byte_to_unicode[Int(byte) + 1])
    end
    return String(take!(out))
end

@testset "Section 26c HF ByteLevel use_regex=false no-split behavior" begin
    text = "Hello there\nHello,   friend!"

    no_regex = KeemenaSubwords._hf_bytelevel_pretokenize_with_spans(
        text;
        add_prefix_space=false,
        use_regex=false,
    )
    @test length(no_regex) == 1
    mapped_piece, span = no_regex[1]
    @test span == (1, ncodeunits(text) + 1)
    raw_piece = try_span_substring(text, span)
    @test raw_piece == text
    @test mapped_piece == _map_bytelevel_piece_no_regex_local(text)

    no_regex_prefixed = KeemenaSubwords._hf_bytelevel_pretokenize_with_spans(
        text;
        add_prefix_space=true,
        use_regex=false,
    )
    @test length(no_regex_prefixed) == 1
    mapped_prefixed, prefixed_span = no_regex_prefixed[1]
    @test prefixed_span == (1, ncodeunits(text) + 1)
    @test mapped_prefixed == _map_bytelevel_piece_no_regex_local(" " * text)
end
