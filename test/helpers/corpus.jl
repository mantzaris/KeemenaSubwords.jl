using TOML

const EDGE_CASE_CORPUS_PATH = joinpath(@__DIR__, "..", "corpus", "tokenization_edge_cases.toml")

function _load_edge_case_corpus_data()::Dict{String,Any}
    isfile(EDGE_CASE_CORPUS_PATH) || error("Missing edge-case corpus file: $(EDGE_CASE_CORPUS_PATH)")
    return TOML.parsefile(EDGE_CASE_CORPUS_PATH)
end

function load_edge_case_corpus_categories(; include_long::Bool=true)::Dict{String,Vector{String}}
    data = _load_edge_case_corpus_data()
    raw_categories = get(data, "categories", Dict{String,Any}())
    categories = Dict{String,Vector{String}}()

    for (name, values_any) in raw_categories
        values_any isa AbstractVector || continue
        categories[String(name)] = String[String(v) for v in values_any]
    end

    if include_long
        generated = get(data, "generated", Dict{String,Any}())
        seed = String(get(generated, "long_seed", "keemena-subwords-long-seed "))
        lengths = Int[Int(x) for x in get(generated, "long_lengths", Int[])]
        long_values = String[]

        if !isempty(seed)
            for n in lengths
                n <= 0 && continue
                reps = cld(n, length(seed))
                full = repeat(seed, reps)
                push!(long_values, full[1:n])
            end
        end

        if !isempty(long_values)
            categories["long_generated"] = long_values
        end
    end

    return categories
end

function load_edge_case_corpus(; include_long::Bool=true)::Vector{String}
    categories = load_edge_case_corpus_categories(; include_long=include_long)
    names = sort!(collect(keys(categories)))
    corpus = String[]
    seen = Set{String}()

    for name in names
        for text in categories[name]
            text in seen && continue
            push!(corpus, text)
            push!(seen, text)
        end
    end

    return corpus
end

function edge_case_corpus_subset(
    n::Integer;
    include_long::Bool=false,
    nonempty_only::Bool=true,
)::Vector{String}
    n <= 0 && return String[]
    corpus = load_edge_case_corpus(; include_long=include_long)
    nonempty_only && (corpus = [s for s in corpus if any(!isspace, s)])
    isempty(corpus) && return String[]
    length(corpus) <= n && return corpus

    idxs = unique(round.(Int, range(1, length(corpus), length=n)))
    return [corpus[i] for i in idxs]
end
