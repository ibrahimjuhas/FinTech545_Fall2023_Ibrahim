corr = [1 0.5 0
    0.5 1 0.5
    0 0.5 1]
sd = [0.2, 0.1, 0.05]
er = [0.05, 0.04, 0.03]

covar = diagm(sd) * corr * diagm(sd)

function optimize_risk(R)
    # R = 0.05

    m = Model(Ipopt.Optimizer)
    set_silent(m)
    # Weights with boundry at 0
    @variable(m, w[i=1:3] >= 0, start = 1 / 3)

    @objective(m, Min, w' * covar * w)
    @constraint(m, sum(w) == 1.0)
    @constraint(m, sum(er[i] * w[i] for i in 1:3) == R)
    optimize!(m)
    #return the objective(risk) as well as the portfolio weights
    return Dict(:risk => objective_value(m), :weights => value.(w), :R => R)
end

returns = [i for i in 0.03:0.001:0.05]
optim_portfolios = DataFrame(optimize_risk.(returns))
plot(sqrt.(optim_portfolios.risk), optim_portfolios.R, legend=:bottomright, label="Efficient Frontier", xlabel="Risk - SD", ylabel="Portfolio Expected Return")

# w = [i for i in 0:.1:1.5]
# returns = .1*w .+ .05*(1 .-w)
# risks = .16*w
# plot(risks, returns, legend=:bottomright, label="", title="Investment A + Rf", xlabel="Risk - SD", ylabel="Portfolio Expected Return")
# scatter!((.16,.1), label="Investment A")

#Sharpe Ratios
optim_portfolios[!, :SR] = (optim_portfolios.R .- 0.03) ./ sqrt.(optim_portfolios.risk)
maxSR = argmax(optim_portfolios.SR)
maxSR_ret = optim_portfolios.R[maxSR]
maxSR_risk = sqrt(optim_portfolios.risk[maxSR])

println("Portfolio Weights at the Maximum Sharpe Ratio: $(optim_portfolios.weights[maxSR])")
println("Portfolio Return : $maxSR_ret")
println("Portfolio Risk   : $maxSR_risk")
println("Portfolio SR     : $(optim_portfolios.SR[maxSR])")






w = [i for i in 0:0.1:2]
returns = maxSR_ret * w .+ 0.03 * (1 .- w)
risks = maxSR_risk * w
plot!(risks, returns, label="", color=:red)
scatter!((maxSR_risk, maxSR_ret), label="Max SR Portfolio")