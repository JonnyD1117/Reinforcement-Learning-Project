include("SPMe_Step.jl")


# using .SPMe_Battery_Model

println("SPMe Step NOT Taken")

# SPMe_Battery_Model.this_print_thing("hello this thing")
xn_old = nothing
xp_old= nothing
xe_old= nothing
Sepsi_p= nothing
Sepsi_n= nothing

I_input= -25.7
init_flag=false

xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag = SPMe_Battery_Model.SPMe_step(xn_old, xp_old, xe_old, Sepsi_p, Sepsi_n, I_input, init_flag)

println("SPMe Step Taken")
println(soc_new[1])
