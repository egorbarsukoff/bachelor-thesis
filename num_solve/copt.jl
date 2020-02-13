using Optim, LinearAlgebra, ForwardDiff

function vec2design(vec)
    @assert(mod(size(vec)[1], 2) == 0)
    len = size(vec)[1]
    n = len ÷ 2
    x = vec[1:len ÷ 2]
    w = vec[len ÷ 2 + 1: len]
    return x, w
end

function design2vec(x, w)
    @assert size(x) == size(w)
    return vcat(x, w)
end

function reduce_design(x, w)
    @assert size(x) == size(w)
    n = size(x, 1)
    concated = [(x[i], w[i]) for i=1:n]
    sort!(concated)
    while true
        changed = false
        for i = 1:(size(concated, 1) - 1)
            if abs(concated[i][1] - concated[i+1][1]) < 1e-2
                concated[i] = ((concated[i][1] + concated[i+1][1])/2,
                    concated[i][2] + concated[i+1][2])
                deleteat!(concated, i+1)
                changed = true
                break
            end
        end
        if !changed
            break
        end
    end
    [concated[i][1] for i=1:1:size(concated, 1)],
        [concated[i][2] for i=1:1:size(concated, 1)]
end

function design2M(f, x, w)
    @assert size(x)[1] == size(w)[1]
    n = size(f(x[1]), 1)
    M = zeros(Float64, n, n)
    for i = 1:size(w)[1]
        t = f(x[i]) * transpose(f(x[i])) * w[i]
        M .+= t
    end
    M
end

function f2min(f, c, x, w)
    w ./= sum(w)
    M = design2M(f, x, w)
    piM = pinv(M)
    if (sum((c - M * piM * c) .^ 2) < 1e-13)
        return (transpose(c) * piM * c)[1]
    else
        return 1e8
    end
end

function find_design(f, c, l_x, h_x)
    n = size(f(l_x), 1)
    @assert n == size(c, 1)

    low_border = vcat(ones(n) .* l_x , zeros(n))
    high_border = vcat(ones(n) .* h_x, ones(n))
    counter = 0
    while true
        init_x = collect(range(l_x, stop = h_x, length=size(c, 1)))
        init_vec = vcat(rand(n) .* (h_x - l_x) .+ l_x, rand(n))
        if counter == 501
            println("Seems like only BFGS not convergence. Try ParticleSwarm+BFGS")
        end
        if counter > 500
            res = optimize((vec -> f2min(poly2, c, vec2design(vec)...)),
                low_border, high_border, init_vec, Fminbox(ParticleSwarm()))
        else
            res = optimize((vec -> f2min(poly2, c, vec2design(vec)...)),
                low_border, high_border, init_vec, Fminbox(BFGS()))
        end


        found_x, found_w = vec2design(res.minimizer)
        found_x, found_w = reduce_design(found_x, found_w)
        @assert size(found_x, 1) == size(found_w, 1)
        idx = filter!(i -> found_w[i] > 1e-7, collect(1:size(found_x, 1)))
        found_x = found_x[idx]
        found_w = found_w[idx]
        found_w ./= sum(found_w)

        err = check(f, c, found_x, found_w, l_x, h_x)

        if err < 1e-5
            println(err)
            return found_x, found_w, res.minimum
        end
        println("One more try err = ", err)
        counter += 1
    end
end

function check(f, c, x, w, l_x, h_x)
    inner_x = filter(v -> min(abs(l_x - v), abs(h_x - v)) > 1e-5, x)
    n = size(x, 1)
    inner_n = size(inner_x, 1)

    fs = [f(v) for v=x]
    A = vcat(transpose.(fs)...,
        transpose.([ForwardDiff.derivative.(f, v) for v=inner_x])...)

    piM = pinv(design2M(f, x, w))
    h = sqrt(transpose(c) * piM * c)


    vars = Array{Array{Float64, 1}, 1}()
    for i in 1:2^(n - 1)
        push!(vars, [i & (1 << (v - 1)) != 0 ? 1 : -1 for v in 1:(n-1)])
    end

    residuals = Array{Float64, 1}()
    for var in vars
        b = vcat([1.], var, zeros(inner_n))
        p = pinv(A) * b
        p_res = sum((A * p - b).^2)
        s = sum([fs[i] * transpose(p) * fs[i] .* w[i] for i=1:n])
        c_res = min(sum((c - s .* h).^2), sum((c + s .* h).^2))
        push!(residuals, c_res+p_res)
    end
    return min(Inf, residuals...)
end


function poly2(x)
    [x, x^2, x^3]
end

function cycle()
    for z in range(0, 1, length=20)
        c = ForwardDiff.derivative.(poly2, z)
        println("z = ", z, "    ", find_design(poly2,c, 0, 1))
    end
end
