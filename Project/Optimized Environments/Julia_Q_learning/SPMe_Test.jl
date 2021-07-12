include("SPMe_Step.jl")

using Plots

# using .SPMe_Battery_Model

println("SPMe Step NOT Taken")

# SPMe_Battery_Model.this_print_thing("hello this thing")
xn = nothing
xp = nothing
xe = nothing
Sepsi_p = nothing
Sepsi_n = nothing

I_input = -25.7
# init_flag=false

num_eps = 1

step_list = []

soc_list =[]
for eps = 1:num_eps
    for step = 1:1800

        init_Flag = (step == 1) ? 1 : 0

        if mod(step, 100) == 0

            println(step)

        end

    xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag = SPMe_Battery_Model.SPMe_step(xn, xp, xe, Sepsi_p, Sepsi_n, I_input, init_Flag)

    global xn = xn_new
    global xp= xp_new
    global xe= xe_new
    global Sepsi_p= Sepsi_p_new
    global Sepsi_n= Sepsi_n_new

    push!(soc_list, soc_new[1])
    push!(step_list, step)
    end
end

# println(xn_old)
# println(step_list)
plot(soc_list)
# plot(step_list)
