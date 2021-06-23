# Add Imports
from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity as SPMe
# from D_SPMe_w_remaining_time_n_soc_states_env import SPMenv as Discrete_SPMe_env
#
# import gym
# from gym import error, spaces, utils, logger
# from gym.utils import seeding
import numpy as np
# import random
# from torch.utils.tensorboard import SummaryWriter
# import logging
import time
from tqdm import tqdm
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


def Discretize_Value(input_val, input_range, num_disc, zipped=False):
    """
    Uniform Discretization of input variable given the min/max range of the input variable and the total number of discretizations desired
    """
    step_size = (input_range[1] - input_range[0]) / num_disc                    # Compute the Average Step-Size for "num_disc" levels
    discrete_values = np.arange(input_range[0], input_range[1], step_size)      #
    index_values = np.arange(0, num_disc, 1)
    zipped_var = zip(index_values, discrete_values)

    if zipped is False:

        for i in index_values:

            if input_val < discrete_values[0]:

                output_val = discrete_values[0]
                output_index = 0

                return output_val, output_index

            elif input_val > discrete_values[-1]:

                output_val = discrete_values[-1]
                output_index = num_disc - 1

                return output_val, output_index

            else:

                if discrete_values[i] <= input_val < discrete_values[i+1]:

                    upper_error = np.abs(discrete_values[i+1] - input_val)
                    lower_error = np.abs(discrete_values[i] - input_val)

                    if lower_error > upper_error:
                        output_val = discrete_values[i+1]
                        output_index = (i + 1)

                    else:
                        output_val = discrete_values[i]
                        output_index = i

                    return output_val, output_index

    else:
        return zipped_var


def epsilon_greedy_policy(Q_table, Q_dims, epsilon):
    num_act = Q_dims[1]
    num_states = Q_dims[0]

    if np.random.uniform(0, 1) <= epsilon:
        action_ind = np.random.randint(0, num_act)

    else:
        ind = np.unravel_index(np.argmax(Q_table, axis=None), Q_table.shape)
        action_ind = ind[1]

    return action_ind

"""
SPMe Battery is +- 1C-rate capped

STATE(S): Battery SOC 
ACTION: Input Current

Discretization: 
    S: 
    A: 
"""


# Initialize Q-Learning Parameters:
# @profile
def main():
    num_avg_runs = 1
    num_episodes = 25000
    episode_duration = 1800

    # Initialize Q-Learning Table
    num_q_states = 10
    num_q_actions = 10

    max_state_val = 1
    min_state_val = 0

    max_action_val = 25.7
    min_action_val = -25.7

    zipped_actions = Discretize_Value(input_val=None, input_range=[min_action_val, max_action_val], num_disc=num_q_actions, zipped=True)

    action_list = {ind:value for ind, value in zipped_actions}

    Q = np.zeros(shape=[num_q_states, num_q_actions])
    alpha = .1
    epsilon = .5
    gamma = .98

    # model = SPMe(init_soc=.5)
    # sim_state_0 = model.full_init_state
    # state_value_0, state_index_0 = Discretize_Value(.5, [0, 1], num_q_states)

    for avg_num in range(num_avg_runs):
        for eps in tqdm(range(num_episodes)):

            # Initialize SPMe Model for each Episode
            SOC_0 = .5
            soc_new = [.5,.5]
            model = SPMe(init_soc=SOC_0)
            # sim_state = sim_state_0
            sim_state = model.full_init_state
            state_value, state_index = Discretize_Value(SOC_0, [0, 1], num_q_states)

            for step in range(episode_duration):
                # Select Action According to Epsilon Greedy Policy
                action_index = epsilon_greedy_policy(Q, [num_q_states, num_q_actions], epsilon)

                # Given the action Index find the corresponding action value
                action = action_list[action_index]

                # soc_0 = soc_new[0] + (1/(25.7*3600))*action

                # Given current state, applying current action via SPMe model
                [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done] = model.SPMe_step(full_sim=True, states=sim_state, I_input=action)
                # soc_1 = None

                # soc_new = [soc_0, soc_1]

                # Update Model's internal states
                sim_state = [bat_states, new_sen_states]

                # Extract the State of Charge & Epsilon Voltage Sensitivity from states
                state_new = soc_new[0].item()

                eps_sensitivity = sensitivity_outputs["dV_dEpsi_sp"]

                # Discretize the Resulting SOC
                new_state_value, new_state_index = Discretize_Value(state_new, [0, 1], num_q_states)

                # Compute Reward Function
                R = eps_sensitivity**2

                # Update Q-Function
                max_Q = np.max(Q[new_state_index][:])

                Q[state_index][action_index] += alpha*(R + gamma*max_Q - Q[state_index][action_index])

                state_value = new_state_value
                state_index = new_state_index

    # np.save("Q_Table", Q)
    #
    # print(Q)


if __name__ == "__main__":

    main()