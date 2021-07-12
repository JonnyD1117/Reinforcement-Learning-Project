
N = 5

thing = Dict( (i,j) => i for i in 1:N for j in i+1:N )

println(thing)

println(thing[(2,4)])
