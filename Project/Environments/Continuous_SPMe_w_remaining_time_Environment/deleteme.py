import gym
import numpy as np
import matplotlib.pyplot as plt
from Continuous_SPMe_w_remaining_time_env import Continuous_SPMenv
# Added this to the Time Based Simulation

from stable_baselines3 import PPO, TD3, DDPG
# from stable_baselines3.common.vec_env.vec_check_nan import VecCheckNan
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.cmd_util import make_vec_env
# # from stable_baselines3.common.utils import set_random_seed
# # from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.ddpg.policies import MlpPolicy

from stable_baselines3.common.noise import NormalActionNoise




model = DDPG.load("./DDPG_Cont_SPMe_Init_Training/model/DDPG_Cont_SPMe_Init_Training_T_1_1_1.pt")

# print(model.actor)
# print(model.critic)
print(model.policy)