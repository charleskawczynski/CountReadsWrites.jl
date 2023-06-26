using Cassette

Cassette.@context CounterCtx;

const read_ops = (Base.getindex, )
const write_ops = (Base.setindex!,)
const read_syms = (:getindex, )
const write_syms = (:setindex!,)
const read_counts = (:getindex, )
const write_counts = (:setindex!,)
const read_tup = collect(zip(read_syms,read_ops,read_counts))
const write_tup = collect(zip(write_syms,write_ops,write_counts))
const all_tup = Iterators.flatten((read_tup, write_tup)) |> collect
const ctypes = ((Array, :array),)

track(::T) where {T} = !isbitstype(T)
const itypes = [Int, Integer, CartesianIndex, AbstractUnitRange, Any]

name(isym::Symbol, csym) = Symbol(:_, isym, :_, csym)
name(isym, csym) = Symbol(:_, Symbol(isym), :_, csym)
name(opsym, isym::Symbol, csym) = Symbol(opsym, :_, isym, :_, csym)
name(opsym, isym, csym) = Symbol(opsym, :_, Symbol(isym), :_, csym)

for ityp in itypes
    for (ctyp, csym) in ctypes
        for (_, op_fun) in write_tup
            @eval function Cassette.prehook(ctx::CounterCtx,
                                            op::typeof($op_fun),
                                            x::$ctyp,
                                            v::Any,
                                            i::$ityp)
                track(x) || return
                $(gen_count(write_ops, name.(write_syms, ityp, csym)))
            end
            @eval function Cassette.prehook(ctx::CounterCtx,
                                            op::typeof($op_fun),
                                            x::$ctyp,
                                            v::Any,
                                            i::$ityp,
                                            j::$ityp)
                track(x) || return
                $(gen_count(write_ops, name.(write_syms, ityp, csym)))
            end
        end

        for (_, op_fun) in read_tup
            @eval function Cassette.prehook(ctx::CounterCtx,
                                            op::typeof($op_fun),
                                            x::$ctyp,
                                            i::$ityp)
                track(x) || return
                $(gen_count(read_ops, name.(read_syms, ityp, csym)))
            end
            @eval function Cassette.prehook(ctx::CounterCtx,
                                            op::typeof($op_fun),
                                            x::$ctyp,
                                            i::$ityp,
                                            j::$ityp)
                track(x) || return
                $(gen_count(read_ops, name.(read_syms, ityp, csym)))
            end
        end
    end
end

for (ctyp, csym) in ctypes
    for (_, op_fun) in write_tup
        @eval function Cassette.prehook(ctx::CounterCtx,
                                        op::typeof($op_fun),
                                        x::$ctyp,
                                        v::Any,
                                        ijk::Vararg{Int,N}) where {N}
            track(x) || return
            $(gen_count(write_ops, name.(write_syms, :Vararg, csym)))
        end
    end

    for (_, op_fun) in read_tup
        @eval function Cassette.prehook(ctx::CounterCtx,
                                        op::typeof($op_fun),
                                        x::$ctyp,
                                        ijk::Vararg{Int,N}) where {N}
            track(x) || return
            $(gen_count(read_ops, name.(read_syms, :Vararg, csym)))
        end
    end
end
