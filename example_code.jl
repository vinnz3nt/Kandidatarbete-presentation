using SymbolicRegression
using MLJ

X = LinRange(0,pi,N)
@. y = sin(X)+cos(X)

model = SRRegressor(
    binary_operators = [+,-,*,/],
    unary_operators = [cos,sin],
    niterations = 100,
    nested_constraints = Dict(
        cos => Dict(cos => 0),
        sin => Dict(sin => 0)
    ),
    maxdepth = 4
)

mach = machine(model,X,y)

fit!(mach)