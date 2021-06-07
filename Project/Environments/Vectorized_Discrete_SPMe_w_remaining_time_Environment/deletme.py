import gym
import numpy as np
import matplotlib.pyplot as plt

# Added this to the Time Based Simulation

# from stable_baselines3 import PPO, TD3, DDPG, DQN
from stable_baselines3 import PPO, TD3, DQN
from stable_baselines3.dqn.policies import MlpPolicy

from stable_baselines3.common.noise import NormalActionNoise
from Discrete_SPMe_w_remaining_time_env import SPMenv as Discrete_SPMe_env

model = DQN.load("./dCse_dEps_Training/model/T_1_1_5_lr_neg3e4_ef_p1.pt")

print(model.q_net)
print(model.q_net_target)