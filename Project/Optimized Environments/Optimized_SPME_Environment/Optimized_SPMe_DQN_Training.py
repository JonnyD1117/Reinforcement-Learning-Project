import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils, logger
from torch.utils.tensorboard import SummaryWriter
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, TD3, DQN
from stable_baselines3.dqn.policies import MlpPolicy
from Optimized_SPMe_env import SPMenv
from time import time_ns


if __name__ == '__main__':

    init_time = time_ns()

    logging_dir_name = "Opt_SPMe_DQN_"
    trial_name = "T_1_1_1"

    env = SPMenv(log_dir=logging_dir_name, log_trial_name=trial_name, log_data=False)




    # HyperParameters
    lr = 3e-4

    # lr = 0.218
    # ef = 0.6827
    ef = .075

    model_name = f"{trial_name}.pt"
    model_path = f"./{logging_dir_name}/model/" + model_name

    # Instantiate Model
    model = DQN(MlpPolicy, env, verbose=1, exploration_fraction=ef, learning_rate=lr)

    # model.load(f"./{logging_dir_name}/model/T_2_1_8.pt", env=env)
    # model = DQN.load(f"./{logging_dir_name}/model/T_2_1_8.pt", env=env)

    # Train OR Load Model
    model.learn(total_timesteps=5000000)

    print(f"total Time Training 1e6 steps: {(time_ns() - init_time)*10**-9}")


    print("TRAINING is OVER")
    env.log_state = False

    model.save(model_path)

    # print("SOC List", env.soc_list)
    epsi_sp_list = []
    cum_eps_sq_list = []
    cum_eps_sq_val = 0
    action_list = []
    soc_list = []
    Concentration_list = []
    Concentration_list1 = []
    max_C_val = np.array([25.67 * 1], dtype=np.float32)
    # action_dict = {0: 1.0 * max_C_val, 1: 0., 2: -1.0 * max_C_val}

    action_list_index = np.arange(-1, 1, .1)
    action_dict = {index: value * max_C_val for index, value in enumerate(action_list_index)}

    obs = env.reset()
    print(f"Initial Observation {obs} ")

    for _ in range(3600):

        action, _states = model.predict(obs, deterministic=True)

        obs, rewards, done, info = env.step(action)

        cum_eps_sq_val += (env.epsi_sp.item(0)) ** 2

        epsi_sp_list.append(env.epsi_sp.item(0))
        cum_eps_sq_list.append(cum_eps_sq_val)
        soc_list.append(env.state_of_charge)
        action_list.append(action_dict[action])

        if done:
            break
            # obs = env.reset()

    plt.figure()
    plt.plot(soc_list)
    plt.title("State of Charge")

    plt.figure()
    plt.plot(epsi_sp_list)
    plt.title("Sensitivity Values")

    plt.figure()
    plt.plot(cum_eps_sq_list)
    plt.title("Cumulative Square Sensitivity Values")

    plt.figure()
    plt.plot(action_list)
    plt.title("Input Currents")

    # plt.figure()
    # plt.plot(soc_list, epsi_sp_list)
    # plt.title("Epsilon Sensitivity vs SOC")
    plt.show()