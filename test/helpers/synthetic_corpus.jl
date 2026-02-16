const _SYNTH_ASCII_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "gamma",
    "hotel", "india", "juliet", "kilo", "lima", "micro", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "xray", "yankee", "zulu",
]

const _SYNTH_UNICODE_WORDS = [
    "café", "naïve", "façade", "coöperate", "你好", "世界", "mañana",
    "über", "🙂", "🚀", "東京", "Δelta",
]

const _SYNTH_PUNCT = [",", ".", "!", "?", ":", ";", "...", "--", "(", ")"]
const _SYNTH_NUMBERS = ["0", "7", "13", "42", "99", "2024", "31415"]

function synthetic_corpus(n::Int)::Vector{String}
    n >= 0 || throw(ArgumentError("synthetic_corpus expects n >= 0"))
    lines = String[]
    n == 0 && return lines

    n_ascii = length(_SYNTH_ASCII_WORDS)
    n_unicode = length(_SYNTH_UNICODE_WORDS)
    n_punct = length(_SYNTH_PUNCT)
    n_numbers = length(_SYNTH_NUMBERS)

    for i in 1:n
        ascii_a = _SYNTH_ASCII_WORDS[mod1(i, n_ascii)]
        ascii_b = _SYNTH_ASCII_WORDS[mod1(3 * i + 1, n_ascii)]
        ascii_c = _SYNTH_ASCII_WORDS[mod1(5 * i + 2, n_ascii)]
        uni_a = _SYNTH_UNICODE_WORDS[mod1(2 * i + 1, n_unicode)]
        uni_b = _SYNTH_UNICODE_WORDS[mod1(7 * i + 3, n_unicode)]
        punct = _SYNTH_PUNCT[mod1(11 * i + 1, n_punct)]
        number = _SYNTH_NUMBERS[mod1(13 * i + 2, n_numbers)]

        pattern = mod1(i, 8)
        line = if pattern == 1
            "$ascii_a $ascii_b$punct $number $uni_a"
        elseif pattern == 2
            "$ascii_a  $ascii_b\t$ascii_c $uni_a $uni_b"
        elseif pattern == 3
            "$ascii_a $number $punct\n$ascii_b $uni_a"
        elseif pattern == 4
            "$uni_a $ascii_a $ascii_b$punct $number"
        elseif pattern == 5
            "$ascii_a($ascii_b) $ascii_c $uni_b $number"
        elseif pattern == 6
            "$ascii_a -- $ascii_b ... $uni_a $number"
        elseif pattern == 7
            "$ascii_a $ascii_b $ascii_c ; $uni_a : $uni_b"
        else
            "$ascii_a\t$ascii_b  $number $punct $uni_a"
        end

        push!(lines, line)
    end

    return lines
end

function synthetic_long_text(corpus)::String
    pieces = String[]
    for line in corpus
        canonical = join(collect(eachsplit(String(line))), " ")
        isempty(canonical) && continue
        push!(pieces, canonical)
    end
    return join(pieces, " ")
end
