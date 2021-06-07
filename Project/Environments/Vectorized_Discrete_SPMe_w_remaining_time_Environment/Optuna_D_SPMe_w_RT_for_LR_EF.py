import optuna
import matplotlib.pyplot as plt
from Discrete_SPMe_w_remaining_time_env import SPMenv
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy


def optimize_agent(trial):

    learning_rate = trial.suggest_float('learning_rate', .00001, .5)
    exploration_fraction = trial.suggest_float("exploration_fraction", .5, .99)

    env = SPMenv(log_data=True, log_dir="Optuna_Hyper_Param_Training", log_trial_name="T_1_1_2")

    model = DQN(MlpPolicy, env, verbose=1,
                learning_rate=learning_rate,
                exploration_fraction=exploration_fraction)

    action_value = {0: -25.67, 1: 0, 2: 25.67}
    model.learn(3000000)

    reward_list = []
    n_episodes = 0
    obs = env.reset()
    while n_episodes < 4:
        for _ in range(3600):

            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)

            reward_list.append(rewards)
            if done:
                n_episodes += 1
                break

    reward_mean = np.mean(reward_list)
    return reward_mean


if __name__ == '__main__':

    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(study_name='Discrete_SPMe_w_RT', sampler=sampler, direction='maximize')
    study.optimize(optimize_agent, n_trials=8, show_progress_bar=True)

    print(study.best_params)  # E.g. {'x': 2.002108042}