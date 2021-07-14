
using LinearAlgebra
using Random
using Plots
using Profile
using ProfileView
using PProf
using BenchmarkTools
# include("SPMe_Step.jl")
#
# using .SPMe_Battery_Model

function SPMe_step(xn_old=nothing, xp_old=nothing, xe_old=nothing, Sepsi_p=nothing, Sepsi_n=nothing, I_input=nothing, init_flag=false)
    #=
    """
    step function runs one iteration of the model given the input current and returns output states and quantities
    States: dict(), I_input: scalar, state_of_charge: scalar
    """
    =#
    SOC_0 = .5

    epsilon_sn = 0.6  # average negative active volume fraction
    epsilon_sp = 0.50  # average positive active volume fraction
    epsilon_e_n = 0.3  # Liquid [electrolyte] volume fraction (pos & neg)
    epsilon_e_p = 0.3

    F = 96487  # Faraday constant
    Rn = 10e-6  # Active particle radius (pose & neg)
    Rp = 10e-6

    R = 8.314  # Universal gas constant
    T = 298.15  # Ambient Temp. (kelvin)

    Ar_n = 1  # Current collector area (anode & cathode)
    Ar_p = 1

    Ln = 100e-6  # Electrode thickness (pos & neg)
    Lp = 100e-6
    Lsep = 25e-6  # Separator Thickness
    Lc = Ln + Lp + Lsep  # Total Cell Thickness

    Ds_n = 3.9e-14  # Solid phase diffusion coefficient (pos & neg)
    Ds_p = 1e-13
    De = 2.7877e-10  # Electrolyte Diffusion Coefficient
    De_p = De
    De_n = De

    kn = 1e-5 / F  # Rate constant of exchange current density (Reaction Rate) [Pos & neg]
    kp = 3e-7 / F

    # Stoichiometric Coef. used for "interpolating SOC value based on OCV Calcs. at 0.0069% and 0.8228%
    stoi_n0 = 0.0069  # Stoich. Coef. for Negative Electrode
    stoi_n100 = 0.6760

    stoi_p0 = 0.8228  # Stoich. Coef for Positive Electrode
    stoi_p100 = 0.442

    SOC = 1  # SOC can change from 0 to 1

    cs_max_n = (3.6e3 * 372 * 1800) / F  # 0.035~0.870=1.0690e+03~ 2.6572e+04
    cs_max_p = (3.6e3 * 274 * 5010) / F  # Positive electrode  maximum solid-phase concentration 0.872~0.278=  4.3182e+04~1.3767e+04

    Rf = 1e-3  #
    as_n = 3 * epsilon_sn / Rn  # Active surface area per electrode unit volume (Pos & Neg)
    as_p = 3 * epsilon_sp / Rp

    Vn = Ar_n * Ln  # Electrode volume (Pos & Neg)
    Vp = Ar_p * Lp

    t_plus = 0.4

    cep = 1000  # Electrolyte Concentration (Assumed Constant?) [Pos & Neg]
    cen = 1000

    # Common Multiplicative Factor use in SS  (Pos & Neg electrodes)
    rfa_n = 1 / (F * as_n * Rn ^ 5)
    rfa_p = 1 / (F * as_p * Rp ^ 5)

    epsi_sep = 1
    epsi_e = 0.3
    epsi_n = epsi_e
    gamak = (1 - t_plus) / (F * Ar_n)

    kappa = 1.1046
    kappa_eff = kappa * (epsi_e ^ 1.5)
    kappa_eff_sep = kappa * (epsi_sep ^ 1.5)

    ####################################################################################################################
    ####################################################################################################################
    # Simulation Settings
    Ts = 1
    simulation_time = 3600
    num_steps = simulation_time // Ts

    # Default Input "Current" Settings
    default_current = 25.67  # Base Current Draw
    ###################################################################################################################
    ###################################################################################################################

    if init_flag == true || xn_old == nothing || xp_old == nothing || xe_old == nothing || Sepsi_p == nothing || Sepsi_n == nothing

        # Initialize the "battery" and 'sensitivity' states (FOR STEP METHOD)
        alpha = SOC_0
        stoi_n = (stoi_n100 - stoi_n0) * alpha + stoi_n0  # Negative Electrode Interpolant
        stoi_p = stoi_p0 - (stoi_p0 - stoi_p100) * alpha  # Positive Electrode Interpolant

        # IF no initial state is supplied to the "step" method, treat step as initial step
        xn_old = [(stoi_n * cs_max_n) / (rfa_n * 10395 * (Ds_n ^ 2)); 0; 0] # stoi_n100 should be changed if the initial soc is not equal to 50 %
        xp_old = [(stoi_p * cs_max_p) / (rfa_p * 10395 * (Ds_p ^ 2)); 0; 0]  # initial positive electrode ion concentration
        xe_old = [0; 0]

        Sepsi_p = [0; 0; 0]
        Sepsi_n = [0; 0; 0]
        Sdsp_p = [0; 0; 0; 0]
        Sdsn_n = [0; 0; 0; 0]

    end

    done_flag = false

    A_dp = [1. 1. 0.; 0. 1. 1.; 0. -0.003465 0.811]
    B_dp = [0.; 0.; -1.]
    C_dp = [7.1823213e-08 8.705844e-06 0.0001450974]

    A_dn = [1. 1. 0.; 0. 1. 1.; 0. -0.0005270265 0.92629]
    B_dn = [0.; 0.; -1.]
    C_dn = [9.1035395451e-09 2.82938292e-06 0.0001209138]

    Ae_dp = [0.964820774248931 0.; 0. 0.964820774248931]
    Be_dp = [6.2185e-06; 6.2185e-06]
    Ce_dp = [-39977.1776789832 0.; 0. 39977.1776789832]

    Sepsi_A_dp = [1. 1. 0.; 0. 1. 1.; 0. -0.003465 0.811]
    Sepsi_B_dp = [0.; 0.; 1.]
    Sepsi_C_dp = [0.00143646294319442 0.174116720387202 2.90194533978671]

    Sepsi_A_dn = [1. 1. 0.; .0 1. 1.; 0. -0.0005270265 0.92629]
    Sepsi_B_dn = [0.; 0.; 1.]

    I = I_input
    Jn = I / Vn
    Jp = -I / Vp

    if Jn == 0

        I = .00000001
        Jn = I / Vn
        Jp = -I / Vp

    end


    # Compute "current timestep" Concentration from "Battery States" via Output Eqn (Pos & Neg)
    yn_new = C_dn * xn_old
    yp_new = C_dp * xp_old
    yep_new = Ce_dp * xe_old

    # println(yn_new[1])
    # println(yp_new[1])

    # Compute "NEXT" time step "Battery States" via State Space Models (Pos & Neg)
    xn_new = A_dn * xn_old + B_dn * Jn
    xp_new = A_dp * xp_old + B_dp * Jp
    xe_new = Ae_dp * xe_old + Be_dp * I

    # Electrolyte Dynamics
    vel = (-I * (0.5 * Lp + 0.5 * Ln) / (Ar_n * kappa_eff) + (-I * Lsep) / (Ar_n * kappa_eff_sep) + (2 * R * T * (1 - t_plus) * (1 + 1.2383) * log((1000 + yep_new[1] / (1000 + yep_new[2])) / F)))

    # Compute "Exchange Current Density" per Electrode (Pos & Neg)
    i_0n = kn * F * (cen * yn_new[1] * (cs_max_n - yn_new[1])) ^ .5
    i_0p = kp * F * (cep * yp_new[1] * (cs_max_p - yp_new[1])) ^ .5

    # Kappa (pos & Neg)
    k_n = Jn / (2 * as_n * i_0n)
    k_p = Jp / (2 * as_p * i_0p)

    # Compute Electrode "Overpotentials"
    eta_n = (R * T * log(k_n + (k_n ^ 2 + 1) ^ 0.5)) / (F * 0.5)
    eta_p = (R * T * log(k_p + (k_p ^ 2 + 1) ^ 0.5)) / (F * 0.5)

    # Record Stoich Ratio (SOC can be computed from this)
    theta_n = yn_new[1] / cs_max_n
    theta_p = yp_new[1] / cs_max_p

    theta = [theta_n, theta_p]  # Stoichiometry Ratio Coefficent

    SOC_n = ((theta_n - stoi_n0) / (stoi_n100 - stoi_n0))
    SOC_p = ((theta_p - stoi_p0) / (stoi_p100 - stoi_p0))

    soc_new = [SOC_n, SOC_p]

    U_n = 0.194 + 1.5 * exp(-120.0 * theta_n) + 0.0351 * tanh((theta_n - 0.286) / 0.083) - 0.0045 * tanh(
        (theta_n - 0.849) / 0.119) - 0.035 * tanh((theta_n - 0.9233) / 0.05) - 0.0147 * tanh(
        (theta_n - 0.5) / 0.034) - 0.102 * tanh((theta_n - 0.194) / 0.142) - 0.022 * tanh(
        (theta_n - 0.9) / 0.0164) - 0.011 * tanh((theta_n - 0.124) / 0.0226) + 0.0155 * tanh(
        (theta_n - 0.105) / 0.029)

    U_p = 2.16216 + 0.07645 * tanh(30.834 - 54.4806 * theta_p) + 2.1581 * tanh(52.294 - 50.294 * theta_p) - 0.14169 * tanh(11.0923 - 19.8543 * theta_p) + 0.2051 * tanh(1.4684 - 5.4888 * theta_p) + 0.2531 * tanh((-theta_p + 0.56478) / 0.1316) -0.02167 * tanh((theta_p - 0.525) / 0.006)

    docv_dCse_n = -1.5 * (120.0 / cs_max_n) * exp(-120.0 * theta_n) + (0.0351 / (0.083 * cs_max_n)) * ((cosh((theta_n - 0.286) / 0.083)) ^ (-2)) - (0.0045 / (cs_max_n * 0.119)) * ((cosh((theta_n - 0.849) / 0.119)) ^ (-2)) - (0.035 / (cs_max_n * 0.05)) * ((cosh((theta_n - 0.9233) / 0.05)) ^ (-2)) - (0.0147 / (cs_max_n * 0.034)) * ((cosh((theta_n - 0.5) / 0.034)) ^ (-2)) - (0.102 / (cs_max_n * 0.142)) * ((cosh((theta_n - 0.194) / 0.142)) ^ (-2)) - (0.022 / (cs_max_n * 0.0164)) * ((cosh((theta_n - 0.9) / 0.0164)) ^ (-2)) - (0.011 / (cs_max_n * 0.0226)) * ((cosh((theta_n - 0.124) / 0.0226)) ^ (-2)) + (0.0155 / (cs_max_n * 0.029)) * ((cosh((theta_n - 0.105) / 0.029)) ^ (-2))

    docv_dCse_p = 0.07645 * (-54.4806 / cs_max_p) * ((1.0 / cosh(30.834 - 54.4806 * theta_p)) ^ 2) + 2.1581 * (-50.294 / cs_max_p) * ((cosh(52.294 - 50.294 * theta_p)) ^ (-2)) + 0.14169 * (19.854 / cs_max_p) * ((cosh(11.0923 - 19.8543 * theta_p)) ^ (-2)) - 0.2051 * (5.4888 / cs_max_p) * ((cosh(1.4684 - 5.4888 * theta_p)) ^ (-2)) - 0.2531 / 0.1316 / cs_max_p * ((cosh((-theta_p + 0.56478) / 0.1316)) ^ (-2)) - 0.02167 / 0.006 /cs_max_p * ((cosh((theta_p - 0.525) / 0.006)) ^ (-2))

    theta_p = theta_p * cs_max_p
    theta_n = theta_n * cs_max_n

    # state space Output Eqn. realization for epsilon_s (Neg & Pos)
    out_Sepsi_p = Sepsi_C_dp * Sepsi_p

    out_Sepsi_p = out_Sepsi_p[1]
    # println("out_Sepsi_p: $(out_Sepsi_p)")
    # state space realization for epsilon_s (Neg & Pos)
    Sepsi_p_new = Sepsi_A_dp * Sepsi_p + Sepsi_B_dp * I  # current input for positive electrode is negative, ... therefore the sensitivity output should be multiplied by -1
    # println("Sepsi_p_new: $(Sepsi_p_new)")
    Sepsi_n_new = Sepsi_A_dn * Sepsi_n + Sepsi_B_dn * I
    # println("Sepsi_n_new: $(Sepsi_n_new)")


    # rho1p_1 = -sign(I) * (-3 * R * T) / (0.5 * F * Rp * as_p) * ((1 + 1 / k_p ^ 2) ^ (-0.5))
    rho1p = R * T / (0.5 * F) * (1 / (k_p + (k_p ^ 2 + 1) ^ 0.5)) * (1 + k_p / ((k_p ^ 2 + 1) ^ 0.5)) * (-3 * Jp / (2 * as_p ^ 2 * i_0p * Rp))

    rho2p = (R * T) / (2 * 0.5 * F) * (cep * cs_max_p - 2 * cep * theta_p) / (cep * theta_p * (cs_max_p - theta_p)) * (1 + (1 / (k_p + .00000001) ^ 2)) ^ (-0.5)

    # rho1n_1 = sign(I) * (-3 * R * T) / (0.5 * F * Rn * as_n) * ((1 + 1 / k_n ^ 2) ^ (-0.5))
    rho1n = R * T / (0.5 * F) * (1 / (k_n + (k_n ^ 2 + 1) ^ 0.5)) * (1 + k_n / ((k_n ^ 2 + 1) ^ 0.5)) * (-3 * Jn / (2 * as_n ^ 2 * i_0n * Rn))

    rho2n = (-R * T) / (2 * 0.5 * F) * (cen * cs_max_n - 2 * cen * theta_n) / (cen * theta_n * (cs_max_n - theta_n)) * (1 + 1 / (k_n + .00000001) ^ 2) ^ (-0.5)

    # sensitivity of epsilon_sp epsilon_sn
    sen_out_spsi_p = (rho1p + (rho2p + docv_dCse_p) * -out_Sepsi_p)

    out_deta_p_desp = rho1p + rho2p * (-1) * out_Sepsi_p
    out_semi_linear_p = docv_dCse_p * out_Sepsi_p

    dV_dEpsi_sp = sen_out_spsi_p

    # Surface Concentration Sensitivity for Epsilon (pos & neg)
    dCse_dEpsi_sp = -1. * out_Sepsi_p * epsi_n

    docv_dCse = [docv_dCse_n, docv_dCse_p]

    V_term = (U_p - U_n) + (eta_p - eta_n) + vel - Rf * I / (Ar_n * Ln * as_n)  # terminal voltage
    R_film = -Rf * I / (Ar_n * Ln * as_n)

    return xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag
end


function Discretization_Dict(input_range, num_disc)

    step_size = (input_range[2] - input_range[1]) / num_disc
    discrete_values = input_range[1]:step_size:input_range[2]
    index_values = 1:1:(num_disc + 1)
    zipped_var = Dict( i => discrete_values[i] for i = 1:1:length(discrete_values))

    return discrete_values, index_values, zipped_var

end


function Discretize_Value(input_val, input_values, row, col)

    input_vect = input_val * ones(row, col)
    minval, argmin = findmin((abs.(input_values - input_vect)))
    output_val = input_values[argmin]
    output_index = argmin

    return output_val, output_index[1]
end


function epsilon_greedy_policy(state_ind, Q_table, Q_dims, epsilon)
    num_act = Q_dims[2]

    prob = rand()

    if prob <= epsilon
        action_ind = rand(1:num_act)
        return action_ind


    else
        action_val, action_ind = findmin(Q_table[state_ind, :])
        return action_ind

    end
end


function q_learning_policy(Q_table, num_states, num_actions, state_range, action_range)
    # Discretization Parameters
    max_state_val = state_range[2]
    min_state_val = state_range[1]

    max_action_val = action_range[2]
    min_action_val = action_range[1]

    SOC_0 = .5
    I_input = -25.7

    soc_state_values, _, _ = Discretization_Dict([min_state_val, max_state_val], num_states)
    action_values, _, _ = Discretization_Dict([min_action_val, max_action_val], num_actions)

    # soc_row, soc_col = size(soc_state_values)
    soc_row = size(soc_state_values)
    soc_row = soc_row[1]
    soc_col = 1

    state_value, state_index = Discretize_Value(SOC_0, soc_state_values, soc_row, soc_col)

    init_flag = 1

    xn = nothing
    xp = nothing
    xe = nothing
    Sepsi_p = nothing
    Sepsi_n = nothing

    action_list = []
    soc_list = []
    push!(soc_list, SOC_0)

    for t = 1:1:1800

        action_value, action_index = findmax(Q_table[state_index,:])

        push!(action_list, action_value)

        I_input = action_values[action_index]

        # action_list = [action_list, I_input]

        if t == 1
            init_flag = 1
            xn = 1.0e+11 * [9.3705; 0; 0]
            xp = 1.0e+11 * [4.5097; 0; 0]
            xe = [0; 0]
            Sepsi_p = [0; 0; 0]
            Sepsi_n = [0; 0; 0]

        else
            init_flag = 0
        end

            xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag = SPMe_step(xn, xp, xe, Sepsi_p, Sepsi_n, I_input, init_flag)

        soc = soc_new[1]

        xn = xn_new
        xp = xp_new
        xe = xe_new
        Sepsi_p = Sepsi_p_new
        Sepsi_n = Sepsi_n_new

        # Discretize the Resulting SOC
        new_state_value, new_state_index = Discretize_Value(soc, soc_state_values, soc_row, soc_col)

        state_value = new_state_value
        state_index = new_state_index
        push!(soc_list, state_value)

    end


    return action_list, soc_list

end


function main_optimized()
    # random.seed(0)

    # Training Duration Parameters
    num_episodes = 50000
    episode_duration = 1800

    # Initialize Q - Learning Table
    # num_q_states = 1000
    # num_q_actions = 101
    num_q_states = 250
    num_q_actions = 11
    # num_q_states = 50000
    # num_q_actions = 500

    # num_q_states = 25
    # num_q_actions = 5

    # Discretization Parameters
    max_state_val = 1
    min_state_val = 0
    max_action_val = 25.7
    min_action_val = -25.7
    # max_action_val = 25.7 * 3
    # min_action_val = -25.7 * 3

    action_values, action_index, action_dict = Discretization_Dict([min_action_val, max_action_val], num_q_actions)
    soc_state_values, soc_state__index, soc_state_dict = Discretization_Dict([min_state_val, max_state_val], num_q_states)


    # Q - Learning Parameters
    Q = zeros(num_q_states+1 , num_q_actions+1)
    alpha = .1
    epsilon = .05
    gamma = .98

    # SPMe Initialization Parameters
    SOC_0 = .5
    I_input = -25.7
    initial_step = 1

    xn = 0
    xp = 0
    xe = 0
    Sepsi_p = 0
    Sepsi_n = 0
    Sdsp_p = 0
    Sdsn_n = 0

    time_list = []

    voltage_list = []
    soc_list = []
    current_list = []

    eps_range = 1:1:episode_duration

    state_size = size(soc_state_values)
    soc_row = state_size[1]
    soc_col = 1

    SOC_0 = rand(num_episodes)

    for eps in 1:num_episodes

        mod(eps, 1000) == 0 ? print("Episode # $(eps) \n") : nothing


        state_value, state_index = Discretize_Value(SOC_0[eps], soc_state_values, soc_row, soc_col)

        for step = 1:1:1800
            # print("Steps = $(step) \n")

            # Select Action According to Epsilon Greedy Policy
            action_index = epsilon_greedy_policy(state_index, Q, [num_q_states, num_q_actions], epsilon)
            # println("Action Index = $(action_index)")

            # Given the action Index find the corresponding action value
            if step == 1
                xn = 1.0e+11 * [9.3705; 0; 0]
                xp = 1.0e+11 * [4.5097; 0; 0]
                xe = [0; 0]
                Sepsi_p = [0; 0; 0]
                Sepsi_n = [0 ; 0; 0]

            end

            initial_step = (step ==1) ? 1 : 0

            I_input = action_values[action_index]

            xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag = SPMe_step(xn, xp, xe, Sepsi_p, Sepsi_n, I_input, initial_step)

            soc = soc_new[1]

            xn = xn_new
            xp = xp_new
            xe = xe_new
            Sepsi_p = Sepsi_p_new
            Sepsi_n = Sepsi_n_new


            # Discretize the Resulting SOC
            new_state_value, new_state_index = Discretize_Value(soc, soc_state_values, soc_row, soc_col)

            # println("New State Value = $(new_state_value)")
            # println("New State Index = $(new_state_index)")

            # Compute Reward Function
            R = dV_dEpsi_sp[1] ^ 2

            # Update Q-Function
            max_Q = maximum(Q[new_state_index, :])

            # println(state_index)
            # println(action_index)

            Q[state_index, action_index] +=  alpha * (R + gamma * max_Q - Q[state_index, action_index])

            state_value = new_state_value
            state_index = new_state_index[1]

            # if soc >= 1.2 || soc < 0 || V_term < 2.25 || V_term >= 4.4
            #     break
            # end
            push!(time_list, step)

        end

    end
    # print(Q)
    # println(time_list)
    # plot(time_list)
    # heatmap(Q)

    return Q

end


# main_optimized()
# @pprof main_optimized()
# @profview main_optimized()
# @time Q_table = main_optimized()
@time Q_table = main_optimized()
num_states = 250
num_actions = 11

state_range = [0, 1]
action_range = [-25.7, 25.7]

action_list, soc_list = q_learning_policy(Q_table, num_states, num_actions, state_range, action_range)

plot!(action_list, title="Action Titles")

# Juno.profiler()
