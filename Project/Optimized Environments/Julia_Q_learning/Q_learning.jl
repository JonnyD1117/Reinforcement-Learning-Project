
using LinearAlgebra
using Random
include("SPMe_Step.jl")

using .SPMe_Battery_Model


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

    return output_val, output_index
end

function epsilon_greedy_policy(state_ind, Q_table, Q_dims, epsilon)
    num_act = Q_dims[2]

    if rand() <= epsilon
        action_ind = rand(1:num_act)

    else
        action_val, action_ind = findmin(Q_table[state_ind, :])
    end

    return action_ind
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

    soc_row, soc_col = size(soc_state_values)
    state_value, state_index = Discretize_Value(SOC_0, soc_state_values, soc_row, soc_col)

    init_flag = 1

    xn = 0
    xp = 0
    xe = 0
    Sepsi_p = 0
    Sepsi_n = 0

    action_list = []
    soc_list = []
    soc_list = [soc_list, SOC_0]

    for t = 1:1:1800

        action_value, action_index = findmax(Q_table[state_index,:])

        I_input = action_values[action_index]

        action_list = [action_list, I_input]

        if t == 1
            init_flag = 1
            xn = 1.0e+11 * [9.3705; 0; 0]
            xp = 1.0e+11 * [4.5097; 0; 0]
            xe = [0; 0]
            Sepsi_p = [0; 0; 0]
            Sepsi_n = [0; 0; 0]
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
        pop!(soc_list, state_value)

    end


    return action_list, soc_list

end


function main_optimized()



    # random.seed(0)

    # Training Duration Parameters
    num_episodes = 1
    episode_duration = 1800

    # Initialize Q - Learning Table
    # num_q_states = 1000
    # num_q_actions = 101
    num_q_states = 250
    num_q_actions = 11
    # num_q_states = 50000
    # num_q_actions = 500

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
    Q = zeros(num_q_states , num_q_actions)
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

    state_size = size(soc_state_values)
    soc_row =state_size[1]
    soc_col = 1

    SOC_0 = rand(num_episodes)

    # init_time = time.time_ns()

    println("Before LOOP")

    for eps = 1:1:num_episodes
        println("In Episode Loop")

        if mod(eps,1000) == 0
            println(eps)
        end

        state_value, state_index = Discretize_Value(SOC_0[eps], soc_state_values, soc_row, soc_col)

        for step = 1:1:episode_duration

            # Select Action According to Epsilon Greedy Policy
            action_index = epsilon_greedy_policy(state_index, Q, [num_q_states, num_q_actions], epsilon)
            # print(f"Action Index = {action_index}")

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
            xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag = SPMe_Battery_Model.SPMe_step(xn, xp, xe, Sepsi_p, Sepsi_n, I_input, initial_step)

            soc = soc_new[1]

            xn = xn_new
            xp = xp_new
            xe = xe_new
            Sepsi_p = Sepsi_p_new
            Sepsi_n = Sepsi_n_new

            # Discretize the Resulting SOC
            new_state_value, new_state_index = Discretize_Value(soc, soc_state_values, soc_row, soc_col)

            # Compute Reward Function
            R = dV_dEpsi_sp[1] ^ 2

            # Update Q-Function
            max_Q = maximum(Q[new_state_index, :])

            Q[state_index, action_index] +=  alpha * (R + gamma * max_Q - Q[state_index, action_index])

            state_value = new_state_value
            state_index = new_state_index

            if soc >= 1.2 || soc < 0 || V_term < 2.25 || V_term >= 4.4
                break
            end

            print(step)

        end

        push!(time_list, step)
    end

    println("OUT of the Loop")
    print(Q)
end



main_optimized()
