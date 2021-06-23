# Add Imports
from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity as SPMe
from D_SPMe_w_remaining_time_n_soc_states_env import SPMenv as Discrete_SPMe_env

import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import logging
import time
import os
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


def epsilon_greedy_policy(state_ind, Q_table, Q_dims, epsilon):
    num_act = Q_dims[1]
    num_states = Q_dims[0]

    if np.random.uniform(0, 1) <= epsilon:
        action_ind = np.random.randint(0, num_act)

    else:
        ind = np.unravel_index(np.argmax(Q_table[state_ind][:]), Q_table.shape)
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
    num_episodes = 150
    episode_duration = 1800

    # Initialize Q-Learning Table
    num_q_states = 100
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

    SOC_0 = .5

    time_list = []

    init_time = time.time_ns()

    for avg_num in range(num_avg_runs):
        for eps in tqdm(range(num_episodes)):

            # os.mkdir(f"./log_files/avg_num_{avg_num}_ep_num_{eps}")
            env = Discrete_SPMe_env(log_dir="", log_trial_name=f"", log_data=False, num_actions=num_q_actions, num_states=num_q_states)

            state_value, state_index = Discretize_Value(SOC_0, [0, 1], num_q_states)

            for step in range(episode_duration):

                # Select Action According to Epsilon Greedy Policy
                action_index = epsilon_greedy_policy(state_index, Q, [num_q_states, num_q_actions], epsilon)

                # Given the action Index find the corresponding action value
                soc, reward, done, _ = env.step(np.array([action_index]))

                # Discretize the Resulting SOC
                new_state_value, new_state_index = Discretize_Value(soc.item(), [0, 1], num_q_states)
                # print(new_state_index)
                # Compute Reward Function
                R = reward

                # Update Q-Function
                max_Q = np.max(Q[new_state_index][:])

                Q[state_index][action_index] += alpha*(R + gamma*max_Q - Q[state_index][action_index])

                state_value = new_state_value
                state_index = new_state_index

                if done is True:
                    break

    final_time = time.time_ns()

    print(f"Total Compute Time: {(final_time - init_time)}")


    # np.save("Q_Table", Q)
    #
    print(Q)

if __name__ == "__main__":

    main()