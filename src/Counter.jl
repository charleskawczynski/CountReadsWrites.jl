abstract type AbstractCounter end
export Counter
@eval mutable struct Counter <: AbstractCounter
    $((:($(name(op[1], itype, csym)) ::Int) for op in all_tup for (_, csym) in ctypes for itype in itypes)...)
end
Counter() = Counter(zeros(length(fieldnames(Counter)))...)

n_reads(c::AbstractCounter) = n_ops(c; match=:getindex)
n_writes(c::AbstractCounter) = n_ops(c; match=:setindex)
function n_ops(c::AbstractCounter; match)
    n = 0
    for fn in fieldnames(typeof(c))
        if occursin(string(match), string(fn))
            n+=getfield(c, fn)
        end
    end
    return n
end
# Relatively inefficient, but there should be no need for performance here...

function Base.show(io::IO, c::AbstractCounter)
    empty_counter = true
    for fn in fieldnames(typeof(c))
        v = getfield(c, fn)
        if v > 0
            println(io, "$fn: $(v)")
            empty_counter = false
        end
    end
    empty_counter && println(io, "Counter is empty")
end

function Base.:(==)(c1::T, c2::T) where {T <: AbstractCounter}
    all(getfield(c1, fn)==getfield(c2, fn) for fn in fieldnames(T))
end

function Base.:(*)(n::Int, c::T) where {T <: AbstractCounter}
    ret = T()
    for fn in fieldnames(T)
        setfield!(ret, fn, n*getfield(c, fn))
    end
    ret
end
