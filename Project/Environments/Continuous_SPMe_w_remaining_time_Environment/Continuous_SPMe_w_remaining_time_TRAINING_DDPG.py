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


if __name__ == '__main__':

    # Instantiate Environment
    logging_dir_name = "DDPG_Cont_SPMe_Init_Training"
    trial_name = "T_1_1_1"

    env = Continuous_SPMenv(log_dir=logging_dir_name, log_trial_name=trial_name, log_data=True)

    # HyperParameters
    lr = 3e-4

    model_name = f"{logging_dir_name}_" + f"{trial_name}.pt"
    model_path = f"./{logging_dir_name}/model/" + model_name

    # Instantiate Model
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=.75 * np.ones(n_actions))
    model = DDPG(MlpPolicy, env, action_noise=action_noise, verbose=1)

    # Train OR Load Model
    model.learn(total_timesteps=100000)

    model.save(model_path)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    epsi_sp_list = []
    action_list = []
    soc_list = []
    Concentration_list = []
    Concentration_list1 = []

    obs = env.reset()
    for _ in range(3600):

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        epsi_sp_list.append(env.epsi_sp.item(0))
        soc_list.append(env.state_of_charge)
        action_list.append(action)

        if done:
            break
            # obs = env.reset()

    plt.figure()
    plt.plot(soc_list)
    plt.show()

    plt.figure()
    plt.plot(epsi_sp_list)
    plt.title("Sensitivity Values")

    plt.figure()
    plt.plot(action_list)
    plt.title("Input Currents")
    plt.show()