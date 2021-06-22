import gym
import numpy as np
import matplotlib.pyplot as plt

# Added this to the Time Based Simulation

# from stable_baselines3 import PPO, TD3, DDPG, DQN
from stable_baselines3 import PPO, TD3, DQN
from stable_baselines3.dqn.policies import MlpPolicy

from stable_baselines3.common.noise import NormalActionNoise
from Large_Discrete_SPMe_w_remaining_time_env import SPMenv as Discrete_SPMe_env


if __name__ == '__main__':
    # Instantiate Environment
    # env_id = 'gym_spm_morestates_discrete_action:spm_morestates_discrete_action-v0'
    # env = gym.make('gym_spm_morestates_discrete_action:spm_morestates_discrete_action-v0')

    logging_dir_name = "Large_D_SPMe_action_space_1_p1_n1"
    trial_name = "T_2_1_666"

    env = Discrete_SPMe_env(log_dir=logging_dir_name, log_trial_name=trial_name)

    # HyperParameters
    lr = 3e-4

    # lr = 0.218
    # ef = 0.6827
    ef = .1

    model_name = f"{trial_name}.pt"
    model_path = f"./{logging_dir_name}/model/" + model_name

    # Instantiate Model
    model = DQN(MlpPolicy, env, verbose=1, exploration_fraction=ef, learning_rate=lr)

    model.load(path="C:/Users/Indy-Windows/Documents/Reinforcement-Learning-Project/Project/Environments/Large_Discrete_SPME_w_remaining_time_Environment/Large_D_SPMe_action_space_1_p1_n1/model/T_2_1_9.pt")

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

    obs = env.reset(test_flag= True)
    print(f"Initial Observation {obs} ")

    for _ in range(3600):

        action, _states = model.predict(obs, deterministic=True)

        obs, rewards, done, info = env.step(action)

        cum_eps_sq_val += (env.epsi_sp.item(0))**2

        epsi_sp_list.append(env.epsi_sp.item(0))
        cum_eps_sq_list.append(cum_eps_sq_val)
        soc_list.append(env.state_of_charge)
        action_list.append(action_dict[action])

        if done:
            break
            # obs = env.reset()
    pred_val = 5


    plt.figure()
    plt.plot(soc_list)
    plt.title("State of Charge")
    plt.savefig(f"./Large_D_SPMe_action_space_1_p1_n1/log_files/T_2_1_9/repeated_prediction/SOC_{pred_val}.png")


    plt.figure()
    plt.plot(epsi_sp_list)
    plt.title("Sensitivity Values")
    plt.savefig(f"./Large_D_SPMe_action_space_1_p1_n1/log_files/T_2_1_9/repeated_prediction/Sensitivity_{pred_val}.png")


    plt.figure()
    plt.plot(cum_eps_sq_list)
    plt.title("Cumulative Square Sensitivity Values")
    plt.savefig(f"./Large_D_SPMe_action_space_1_p1_n1/log_files/T_2_1_9/repeated_prediction/Cum_Sen_{pred_val}.png")


    plt.figure()
    plt.plot(action_list)
    plt.title("Input Currents")
    plt.savefig(f"./Large_D_SPMe_action_space_1_p1_n1/log_files/T_2_1_9/repeated_prediction/InputCurrent_{pred_val}.png")

    # plt.figure()
    # plt.plot(soc_list, epsi_sp_list)
    # plt.title("Epsilon Sensitivity vs SOC")
    # plt.show()








#    logging_dir_name = "Large_D_SPMe_action_space_1_p1_n1"
#     trial_name = "T_2_1_666"
#
# model_name = f"{trial_name}.pt"
# model_path = f"./{logging_dir_name}/model/" + model_name






# plt.savefig(f"./30_Million_Training/model/images/REPEAT_w_time_remaining_1T{trial_num}_stoich_{stoich_val}_SOC.png")
# np.save(f"./30_Million_Training/model/outputs/REPEAT_w_time_remaining_1T{trial_num}_stoich_{stoich_val}_SOC", soc_list)
# plt.savefig(f"./30_Million_Training/model/images/REPEAT_w_time_remaining_1T{trial_num}_stoich_{stoich_val}_remaining_time.png")
# np.save(f"./30_Million_Training/model/outputs/REPEAT_w_time_remaining_1T{trial_num}_stoich_{stoich_val}_SOC", remaining_time_list)
# plt.savefig(f"./30_Million_Training/model/images/REPEAT_w_time_remaining_1T{trial_num}_stoich_{stoich_val}_input_current.png")
# np.save(f"./30_Million_Training/model/outputs/REPEAT_w_time_remaining_1T{trial_num}_stoich_{stoich_val}_SOC", action_list)


