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

include("../library/bt_american.jl")
include("../library/ewCov.jl")
include("../library/expost_factor.jl")
include("../library/fitted_model.jl")
include("../library/gbsm.jl")
include("../library/missing_cov.jl")
include("../library/return_calculate.jl")
include("../library/return_accumulate.jl")
include("../library/RiskStats.jl")
include("../library/simulate.jl")

#Test 1 - missing covariance calculations
#Generate some random numbers with missing values.
function generate_with_missing(n,m; pmiss=.25)
    x = Array{Union{Missing,Float64},2}(undef,n,m)

    for i in 1:n, j in 1:m
        if rand() >= pmiss
            x[i,j] = randn()
        end
    end
    return x
end

Random.seed!(2)
x = generate_with_missing(10,5,pmiss=.2)
CSV.write("data/test1.csv",DataFrame(x,:auto))

x = CSV.read("data/test1.csv",DataFrame)
#1.1 Skip Missing rows - Covariance
cout = missing_cov(Matrix(x),skipMiss=true)
CSV.write("data/testout_1.1.csv",DataFrame(cout,:auto))
#1.2 Skip Missing rows - Correlation
cout = missing_cov(Matrix(x),skipMiss=true,fun=cor)
CSV.write("data/testout_1.2.csv",DataFrame(cout,:auto))
#1.3 Pairwise - Covariance
cout = missing_cov(Matrix(x),skipMiss=false)
CSV.write("data/testout_1.3.csv",DataFrame(cout,:auto))
#1.2 Pairwise - Correlation
cout = missing_cov(Matrix(x),skipMiss=false,fun=cor)
CSV.write("data/testout_1.4.csv",DataFrame(cout,:auto))

#Test 2 - EW Covariance
Random.seed!(3)
x = generate_with_missing(40,5,pmiss=0.0)
CSV.write("data/test2.csv",DataFrame(x,:auto))

x = CSV.read("data/test2.csv",DataFrame)
#2.1 EW Covariance λ=0.97
cout = ewCovar(Matrix(x),0.97)
CSV.write("data/testout_2.1.csv",DataFrame(cout,:auto))
#2.2 EW Correlation λ=0.94
cout = ewCovar(Matrix(x),0.94)
sd = 1 ./ sqrt.(diag(cout))
cout = diagm(sd) * cout * diagm(sd)
CSV.write("data/testout_2.2.csv",DataFrame(cout,:auto))
#2.3 EW Cov w/ EW Var(λ=0.94) EW Correlation(λ=0.97)
cout = ewCovar(Matrix(x),0.97)
sd1 = sqrt.(diag(cout))
cout = ewCovar(Matrix(x),0.94)
sd = 1 ./ sqrt.(diag(cout))
cout = diagm(sd1) * diagm(sd) * cout * diagm(sd) * diagm(sd1)
CSV.write("data/testout_2.3.csv",DataFrame(cout,:auto))

#Test 3 - non-psd matrices

#3.1 near_psd covariance
cin = CSV.read("data/testout_1.3.csv",DataFrame)
cout = near_psd(Matrix(cin))
CSV.write("data/testout_3.1.csv",DataFrame(cout,:auto))

#3.2 near_psd Correlation
cin = CSV.read("data/testout_1.4.csv",DataFrame)
cout = near_psd(Matrix(cin))
CSV.write("data/testout_3.2.csv",DataFrame(cout,:auto))

#3.3 Higham covariance
cin = CSV.read("data/testout_1.3.csv",DataFrame)
cout = higham_nearestPSD(Matrix(cin))
CSV.write("data/testout_3.3.csv",DataFrame(cout,:auto))

#3.2 Higham Correlation
cin = CSV.read("data/testout_1.4.csv",DataFrame)
cout = higham_nearestPSD(Matrix(cin))
CSV.write("data/testout_3.4.csv",DataFrame(cout,:auto))

#4 cholesky factorization
cin = Matrix(CSV.read("data/testout_3.1.csv",DataFrame))
n,m = size(cin)
cout = zeros(Float64,(n,m))
chol_psd!(cout,cin)
CSV.write("data/testout_4.1.csv",DataFrame(cout,:auto))


#5 Normal Simulation

Random.seed!(4)
cin = fill(0.75,(5,5)) + diagm(fill(0.25,5))
sd = 0.1 * randn(5).^2
cin = sd' .* cin .* sd
CSV.write("data/test5_1.csv",DataFrame(cin,:auto))
cin = fill(0.75,(5,5)) + diagm(fill(0.25,5))
cin[1,2] = 1
cin[2,1] = 1
cin = sd' .* cin .* sd
CSV.write("data/test5_2.csv",DataFrame(cin,:auto))
cin = fill(0.75,(5,5)) + diagm(fill(0.25,5))
cin[1,2] = 0
cin[2,1] = 0
cin = sd' .* cin .* sd
CSV.write("data/test5_3.csv",DataFrame(cin,:auto))

#5.1 PD Input
cin = CSV.read("data/test5_1.csv",DataFrame) |> Matrix
cout = cov(simulateNormal(100000, cin))
CSV.write("data/testout_5.1.csv",DataFrame(cout,:auto))

# 5.2 PSD Input
cin = CSV.read("data/test5_2.csv",DataFrame) |> Matrix
cout = cov(simulateNormal(100000, cin))
CSV.write("data/testout_5.2.csv",DataFrame(cout,:auto))

# 5.3 nonPSD Input, near_psd fix
cin = CSV.read("data/test5_3.csv",DataFrame) |> Matrix
cout = cov(simulateNormal(100000, cin,fixMethod=near_psd))
CSV.write("data/testout_5.3.csv",DataFrame(cout,:auto))

# 5.4 nonPSD Input Higham Fix
cin = CSV.read("data/test5_3.csv",DataFrame) |> Matrix
cout = cov(simulateNormal(100000, cin,fixMethod=higham_nearestPSD))
CSV.write("data/testout_5.4.csv",DataFrame(cout,:auto))

# 5.5 PSD Input - PCA Simulation
cin = CSV.read("data/test5_2.csv",DataFrame) |> Matrix
cout = cov(simulate_pca(cin,100000,pctExp=.99))
CSV.write("data/testout_5.5.csv",DataFrame(cout,:auto))

# Test 6

# 6.1 Arithmetic returns
prices = CSV.read("data/test6.csv",DataFrame)
rout = return_calculate(prices,dateColumn="Date")
CSV.write("data/test6_1.csv",rout)

# 6.2 Log returns
prices = CSV.read("data/test6.csv",DataFrame)
rout = return_calculate(prices,method="LOG", dateColumn="Date")
CSV.write("data/test6_2.csv",rout)