module CountReadsWrites

isequal_op(::T, ::T) where {T} = true
isequal_op(a, b) = error("oops")

function gen_count(ops, names)
    body = Expr(:block)
    for (op, name) in zip(ops, names)
        e = quote
            if isequal_op(op, $op)
                ctx.metadata.$name += 1
            end
        end
        push!(body.args, e)
    end
    body
end

export @count_reads_writes

include("overdub.jl")
include("Counter.jl")
include("count_reads_writes.jl")

end # module CountReadsWrites
