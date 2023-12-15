using CSV
using Distributions
using Plots
using StatsPlots
using QuadGK
using DataFrames
using Ipopt
using JuMP
using LoopVectorization
using StatsBase
using LinearAlgebra
using Random

#Problem #1
# sd = .05
# nu = 5
# ts = sqrt( sd^2*(nu-2)/nu )
# td = TDist(5)*ts

# x = rand(td,500)
# CSV.write("Project/problem1.csv",DataFrame(:x=>x))

x = CSV.read("Project/problem1.csv",DataFrame).x

include("../../library/fitted_model.jl")
include("../../library/RiskStats.jl")
include("../../library/simulate.jl")
include("../../library/return_calculate.jl")

# include("../fitted_model.jl")
# include("../RiskStats.jl")

tFit = fit_general_t(x)
nFit = fit_normal(x)

nVaR = VaR(nFit.errorModel; alpha=.05)
tVaR = VaR(tFit.errorModel; alpha=.05)

nES = ES(nFit.errorModel,alpha=0.05)
tES = ES(tFit.errorModel; alpha=0.05)

println("Normal  : VaR $(nVaR) ES $(nES)")
println("Fitted T: VaR $(tVaR) ES $(tES)")

# Normal  : VaR 0.08133532747633611 ES 0.10177426982201689
# Fitted T: VaR 0.07647576841291588 ES 0.11321759172396519

minX, maxX = extrema(x)
df = DataFrame(:x=>[i for i in minX:.001:maxX])
df[!,:tPDF] = pdf.(tFit.errorModel,df.x)
df[!,:nPDF] = pdf.(nFit.errorModel,df.x)

plot(df.x,df.nPDF, label="",color=:red)
plot!(df.x,df.tPDF, label="",color=:blue)
vline!([-nVaR], label="Normal VaR",color=:red)
vline!([-tVaR],label="T VaR",color=:blue)
vline!([-nES], label="Normal ES",color=:red)
vline!([-tES],label="T ES",color=:blue)

# Normal  : VaR 0.08133532747633611 ES 0.10177426982201689
# Fitted T: VaR 0.07647576841291577 ES 0.11321759172396459



#Problem 3
# include("../simulate.jl")
# include("../../Week04/return_calculate.jl")
portfolio = CSV.read("Project/portfolio.csv",DataFrame)
prices = CSV.read("Project/DailyPrices.csv",DataFrame)

#filter portfolio for testing
# portfolio = portfolio[
#                 [portfolio.Stock[i] ∈ ["AAPL", "IBM"] for i in 1:length(portfolio.Stock)]
#                     ,:]

#current Prices
current_prices = prices[size(prices,1),:]


#discrete returns
returns = return_calculate(prices,dateColumn="Date")

stocks = names(returns)
intersect!(stocks,portfolio.Stock)

fittedModels = Dict{String,FittedModel}()

for stock in stocks
    fittedModels[stock] = fit_general_t(returns[!,stock])
end

#construct the copula:
#Start the data frame with the U of the SPY - we are assuming normallity for SPY
U = DataFrame()
for nm in stocks
    U[!,nm] = fittedModels[nm].u
end

R = corspearman(Matrix(U))


#what's the rank of R
evals = eigvals(R)
if min(evals...) > -1e-8
    println("Matrix is PSD")
else
    println("Matrix is not PSD")
end

#simulation
NSim = 5000
simU = DataFrame(
            #convert standard normals to U
            cdf(Normal(),
                simulate_pca(R,NSim)  #simulation the standard normals
            )   
            , stocks
        )

simulatedReturns = DataFrame()
for stock in stocks
    simulatedReturns[!,stock] = fittedModels[stock].eval(simU[!,stock])
end


#Protfolio Valuation
iteration = [i for i in 1:NSim]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

nVals = size(values,1)
currentValue = Vector{Float64}(undef,nVals)
simulatedValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
for i in 1:nVals
    price = current_prices[values.Stock[i]]
    currentValue[i] = values.Holding[i] * price
    simulatedValue[i] = values.Holding[i] * price*(1.0+simulatedReturns[values.iteration[i],values.Stock[i]])
    pnl[i] = simulatedValue[i] - currentValue[i]
end
values[!,:currentValue] = currentValue
values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl

values[!,:Portfolio] = convert.(String,values.Portfolio)

risk = aggRisk(values,[:Portfolio])
# temp = aggRisk(values,Vector{Symbol}(undef,0))
# temp[!,:Portfolio] .= "Total"
# risk = vcat(risk,temp)

CSV.write("Project/problem3_out.csv", risk)

# julia> println(risk)
# 4×14 DataFrame
#  Row │ Portfolio  currentValue  VaR95    ES95     VaR99    ES99           Standard_Dev  min             max            mean      VaR95_Pct  VaR99_Pct  ES95_Pct   ES99_Pct  
#      │ String     Float64       Float64  Float64  Float64  Float64        Float64       Float64         Float64        Float64   Float64    Float64    Float64    Float64   
# ─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │ A             1.08932e6  20186.5  27883.9  32343.0  40640.7            12719.4   -71385.8        75130.9        434.144   0.0185314  0.0296911  0.0255977  0.0373085
#    2 │ B             5.74542e5  11935.2  16146.7  18589.4  23407.6             7326.49  -46698.4        33991.3        -56.7041  0.0207734  0.0323552  0.0281035  0.0407412
#    3 │ C             1.38741e6  26527.7  34413.2  38635.6  48405.0            16381.7       -1.21282e5  99147.6        567.453   0.0191203  0.0278473  0.0248039  0.0348887
#    4 │ Total         3.05127e6  55476.6  75056.4  85131.9      1.06542e5      35018.0       -2.3667e5       1.78344e5  944.893   0.0181815  0.0279005  0.0245984  0.0349172