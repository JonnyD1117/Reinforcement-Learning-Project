from Optimized_SPMe_STEP import SPMe_step
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import cProfile, pstats, io


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def Discretization_Dict(input_range, num_disc):

    step_size = (input_range[1] - input_range[0]) / num_disc
    discrete_values = np.arange(input_range[0], input_range[1], step_size)
    index_values = np.arange(1, (num_disc + 1), 1)
    zipped_var = {i: discrete_values[i] for i in range(len(discrete_values))}

    return [discrete_values, index_values, zipped_var]


def Discretize_Value(input_val, input_values, row, col):

    input_vect = input_val * np.ones((row, col))
    argmin = np.argmin(np.abs(input_values - input_vect))
    output_val = input_values[argmin]
    output_index = argmin

    return [output_val, output_index]


def epsilon_greedy_policy(state_ind, Q_table, Q_dims, epsilon):
    num_act = Q_dims[1]

    if np.random.uniform(0, 1) <= epsilon:
        action_ind = np.random.randint(1, (num_act))

    else:
        action_ind = np.argmax(Q_table[state_ind, :])

    return action_ind


def q_learning_policy(Q_table, num_states, num_actions, state_range, action_range):

    # Discretization Parameters
    max_state_val = state_range[1]
    min_state_val = state_range[0]

    max_action_val = action_range[1]
    min_action_val = action_range[0]

    SOC_0 = .5
    I_input = -25.7

    [soc_state_values, _, _] = Discretization_Dict([min_state_val, max_state_val], num_states)
    [action_values, _, _] = Discretization_Dict([min_action_val, max_action_val], num_actions)

    # print(soc_state_values)
    # print(np.size(soc_state_values))
    # soc_row, soc_col = np.size(soc_state_values)

    soc_shape = np.shape(soc_state_values)
    # print(soc_shape)
    soc_row = soc_shape[0]
    soc_col = 1
    [state_value, state_index] = Discretize_Value(SOC_0, soc_state_values, soc_row, soc_col)

    init_flag = 1

    xn = 0
    xp = 0
    xe = 0
    Sepsi_p = 0
    Sepsi_n = 0

    action_list = []
    soc_list = []
    soc_list.append(SOC_0)

    for t in range(1, 1800):

        # print(t)

        action_index = np.argmax(Q_table[state_index][:])

        # print(action_index)

        I_input = action_values[action_index]
        action_list.append(I_input)

        if t == 1:
            init_flag = 1
            xn = 1.0e+11 * np.array([[9.3705], [0], [0]])
            xp = 1.0e+11 * np.array([[4.5097], [0], [0]])
            xe = np.array([[0], [0]])
            Sepsi_p = np.array([[0], [0], [0]])
            Sepsi_n = np.array([[0], [0], [0]])

            [xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag] = SPMe_step(xn, xp, xe, Sepsi_p, Sepsi_n, I_input, init_flag);

        soc = soc_new[0]

        xn = xn_new
        xp = xp_new
        xe = xe_new
        Sepsi_p = Sepsi_p_new
        Sepsi_n = Sepsi_n_new

        # Discretize the Resulting SOC
        [new_state_value, new_state_index] = Discretize_Value(soc, soc_state_values, soc_row, soc_col)

        state_value = new_state_value
        state_index = new_state_index

        soc_list.append(state_value)

    return [action_list, soc_list]


def main_optimized():

    np.random.seed(0)

    # Training Duration Parameters
    num_episodes = 100000
    episode_duration = 1800

    # Initialize Q - Learning Table
    # num_q_states = 1000
    # num_q_actions = 101
    # num_q_states = 250
    # num_q_actions = 11
    num_q_states = 101
    num_q_actions = 45

    # num_q_states = 50000
    # num_q_actions = 500

    # Discretization Parameters
    max_state_val = 1
    min_state_val = 0
    # max_action_val = 25.7
    # min_action_val = -25.7
    max_action_val = 25.7 * 3
    min_action_val = -25.7 * 3

    [action_values, action_index, action_dict] = Discretization_Dict([min_action_val, max_action_val], num_q_actions)
    [soc_state_values, soc_state__index, soc_state_dict] = Discretization_Dict([min_state_val, max_state_val], num_q_states)

    # Q - Learning Parameters
    Q = np.zeros([num_q_states, num_q_actions])
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

    # [soc_row, soc_col] = np.size(soc_state_values)
    soc_row = np.shape(soc_state_values)[0]
    soc_col = 1

    # SOC_0 = np.random.uniform(0, 1, size=num_episodes)

    SOC_0 = .5

    init_time = time.time_ns()

    epsilon_list = np.linspace(.6, .01, num_episodes)
    alpha_list = np.linspace(.5, .05, num_episodes)

    for eps in tqdm(range(1, num_episodes)):

        epsilon = epsilon_list[eps]
        alpha = alpha_list[eps]
        if eps % 1000 == 0:
            print(eps)

        # [state_value, state_index] = Discretize_Value(SOC_0[eps], soc_state_values, soc_row, soc_col)
        [state_value, state_index] = Discretize_Value(SOC_0, soc_state_values, soc_row, soc_col)

        for step in range(1, episode_duration):
            # Select Action According to Epsilon Greedy Policy
            action_index = epsilon_greedy_policy(state_index, Q, [num_q_states, num_q_actions], epsilon)
            # print(f"Action Index = {action_index}")

            # Given the action Index find the corresponding action value
            if step == 1:
                xn = 1.0e+11 * np.array([[9.3705], [0], [0]])
                xp = 1.0e+11 * np.array([[4.5097], [0], [0]])
                xe = np.array([[0], [0]])
                Sepsi_p = np.array([[0], [0], [0]])
                Sepsi_n = np.array([[0], [0], [0]])

            initial_step = 1 if step == 1 else 0

            # print(action_values)
            # print(len(action_values))

            I_input = action_values[action_index]

            [xn_new, xp_new, xe_new, Sepsi_p_new, Sepsi_n_new, dV_dEpsi_sp, soc_new, V_term, theta, docv_dCse, done_flag] =\
                SPMe_step(xn, xp, xe, Sepsi_p, Sepsi_n, I_input=I_input, init_flag=initial_step)

            soc = soc_new[0]

            xn = xn_new
            xp = xp_new
            xe = xe_new
            Sepsi_p = Sepsi_p_new
            Sepsi_n = Sepsi_n_new

            # Discretize the Resulting SOC
            [new_state_value, new_state_index] = Discretize_Value(soc, soc_state_values, soc_row, soc_col)

            # Compute Reward Function
            R = dV_dEpsi_sp[0] ** 2

            # Update Q-Function
            max_Q = np.max(Q[new_state_index, :])

            Q[state_index, action_index] += alpha * (R + gamma * max_Q - Q[state_index, action_index])

            state_value = new_state_value
            state_index = new_state_index

            if soc > 1.0 or soc < 0 or V_term < 2.25 or V_term >= 4.4:
                break

        # time_list = [time_list, step]

    final_time = time.time_ns()

    print(f"Total Episode Time: {final_time - init_time} ")

    soc_range = [0,1]
    action_range = [-25.7*3, 25.7*3]

    [action_list, soc_list] = q_learning_policy(Q, num_q_states, num_q_actions, [0, 1], [-25.7*3, 25.7*3])

    # print(f"ACTION: {np.shape(action_list)}")
    # print(f"SOC: {np.shape(soc_list)}")

    plt.figure()
    plt.plot(soc_list)
    plt.title("SOC")
    plt.figure()
    plt.plot(action_list)
    plt.title("Input Current")
    plt.show()

if __name__ == "__main__":

    # main()

    main_optimized()

