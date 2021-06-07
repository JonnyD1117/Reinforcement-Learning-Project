import gym
import numpy as np
import matplotlib.pyplot as plt

# Added this to the Time Based Simulation

# from stable_baselines3 import PPO, TD3, DDPG, DQN
from stable_baselines3 import PPO, TD3, DQN
from stable_baselines3.dqn.policies import MlpPolicy

from stable_baselines3.common.noise import NormalActionNoise
from Discrete_SPMe_w_remaining_time_env import SPMenv as Discrete_SPMe_env


if __name__ == '__main__':
    # Instantiate Environment
    # env_id = 'gym_spm_morestates_discrete_action:spm_morestates_discrete_action-v0'
    # env = gym.make('gym_spm_morestates_discrete_action:spm_morestates_discrete_action-v0')

    logging_dir_name = "dCse_dEps_Training"
    trial_name = "T_1_1_5_lr_neg3e4_ef_p1"

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

    # Train OR Load Model
    model.learn(total_timesteps=50000000)

    print("TRAINING is OVER")
    env.log_state = False

    model.save(model_path)

    # model = DQN.load(model_path)

    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    #
    # print("Mean Reward = ", mean_reward)


    # print("SOC List", env.soc_list)

    epsi_sp_list = []
    action_list = []
    soc_list = []
    Concentration_list = []
    Concentration_list1 = []
    max_C_val = np.array([25.67 * 1], dtype=np.float32)
    action_dict = {0: 1.0 * max_C_val, 1: 0., 2: -1.0 * max_C_val}

    obs = env.reset(test_flag= True)
    for _ in range(3600):

        action, _states = model.predict(obs, deterministic=True)

        obs, rewards, done, info = env.step(action)

        epsi_sp_list.append(env.epsi_sp.item(0))
        soc_list.append(env.state_of_charge)
        action_list.append(action)

        if done:
            break
            # obs = env.reset()

    plt.figure()
    plt.plot(soc_list)
    plt.title("State of Charge")
    plt.show()

    plt.figure()
    plt.plot(epsi_sp_list)
    plt.title("Sensitivity Values")

    plt.figure()
    plt.plot(action_list)
    plt.title("Input Currents")
    plt.show()
