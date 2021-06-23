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
from tqdm import tqdm
import cProfile, pstats, io
import os
import matplotlib.pyplot as plt

avg_num = 1
eps = 1
max_C_val = 25.7


# os.mkdir(f"./log_files/avg_num_{avg_num}_ep_num_{eps}")


# action_list = np.arange(-1, 1, .1)
# action_dict = {index: value * max_C_val for index, value in enumerate(action_list)}
# print(action_dict)

soc_list = []

env = Discrete_SPMe_env(log_dir="", log_trial_name=f"avg_num_{avg_num}_ep_num_{eps}", log_data=False, num_actions=20, num_states=None)

for _ in range(1800):


    soc, reward, done, _ = env.step(np.array([0]))
    soc_list.append(soc)

plt.figure()
plt.plot(soc_list)
plt.show()