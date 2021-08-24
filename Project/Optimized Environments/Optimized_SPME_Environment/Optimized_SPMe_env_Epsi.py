from Optimized_SPMe_w_Sensitivity_Params import Opt_SPMe_Model
import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import logging
import time
import matplotlib.pyplot as plt
import cProfile


class SPMenv(gym.Env):

    def __init__(self, time_step=1, training_duration=1800, log_dir=None, log_trial_name=None, log_data=False, SOC=.5):

        self.global_counter = 0
        self.episode_counter = 0
        self.time_horizon_counter = 0
        self.training_duration = training_duration
        self.log_state = log_data

        # if self.log_state is True:
        #     if log_dir is None:
        #         print("NO Logging Directory Name Given")
        #         exit()
        #
        #     elif log_trial_name is None:
        #         print("NO Logging Trial Name Given")
        #         exit()
        #
        #     else:
        #         self.writer = SummaryWriter(f'./{log_dir}/log_files/{log_trial_name}')
        # logging.basicConfig(filename=f'./{log_dir}/log_files/{log_trial_name}/{log_trial_name}.log', level=logging.INFO)

        self.soc_list = []

        self.cs_max_n = (3.6e3 * 372 * 1800) / 96487
        self.cs_max_p = (3.6e3 * 274 * 5010) / 96487

        self.time_step = time_step
        self.start_time = time.ctime(time.time())
        self.dt = 1
        self.step_counter = 0
        self.model = Opt_SPMe_Model(init_soc=SOC)

        upper_state_limits = np.array([np.inf], dtype=np.float32)
        lower_state_limits = np.array([-np.inf], dtype=np.float32)
        max_C_val = np.array([25.67*3], dtype=np.float32)

        self.SOC_0 = SOC
        self.state_of_charge = SOC
        self.epsi_sp = None
        self.term_volt = None

        self.sen_list = []
        self.sen_sqr_list = []

        self.min_soc = .04
        self.max_soc = 1.
        self.min_term_voltage = 2.74
        self.max_term_voltage = 4.1

        # self.action_dict = {0: 1.0*max_C_val, 1: np.array([0.0], dtype=np.float32), 2: -1.0*max_C_val}
        action_list = np.arange(-1, 1, .1)
        self.action_space = spaces.Discrete(len(action_list))
        self.action_dict = {index: value*max_C_val for index, value in enumerate(action_list)}

        self.observation_space = spaces.Box(lower_state_limits, upper_state_limits, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.sim_states = None

        self.steps_beyond_done = None
        self.state_output = None

        # TensorBoard Variables
        self.tb_C_se0 = None
        self.tb_C_se1 = None
        self.tb_epsi_sp = None
        self.tb_input_current = None
        self.tb_state_of_charge = SOC
        self.tb_state_of_charge_1 = SOC
        self.tb_term_volt = None
        self.tb_dCs_deps = None

        self.tb_reward_list = []
        self.cumulative_reward = 0
        self.tb_reward_mean = 0
        self.tb_reward_mean_counter = 1
        self.tb_reward_sum = 0
        self.tb_instantaneous_reward = 0
        self.tb_remaining_time = training_duration

        self.rec_epsi_sp = []
        self.rec_input_current = []
        self.rec_state_of_charge = []
        self.rec_term_volt = []
        self.rec_time = []

        self.time = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_time(self):
        total_time = self.time_step*self.time_horizon_counter
        return total_time

    @staticmethod
    def reward_function(sensitivity_value):

        reward = (1.0*sensitivity_value)**2
        return reward

    def increment_mean(self, new_value, prev_mean, mean_counter):

        if mean_counter == 0:
            new_mean = prev_mean

        else:
            new_mean = prev_mean + ((new_value - prev_mean) / mean_counter)

        return new_mean

    def step(self, action):

        action = action.item()                          # Extract Action Index
        input_current = self.action_dict[action]        # Obtain Action Value

        if self.step_counter == 0:                      # If STEP Counter == 0: Set Init Flag
            init_state_flag = True
        else:                                           # Else: Do not set Init Flag
            init_state_flag = False

        self.step_counter += 1                          # Increment Step Counter

        # Compute New Battery & Sensivitiy States
        new_states, outputs, sensitivity, soc_new, theta, V_term = self.model.SPMe_step(states=self.sim_states, I_input=input_current, init_Flag=init_state_flag)

        self.sim_states = new_states                    # Set Simulation States (Required for propigating SPMe model)
        self.state = [soc_new[1].item()]                  # Set Environment State

        self.soc_list.append(soc_new[1].item())         #
        self.state_of_charge = soc_new[1].item()        #
        self.state_output = outputs                     #

        self.epsi_sp = sensitivity['dV_dEpsi_sp']       # Set Epsilon Sensitivity

        self.sen_list.append(self.epsi_sp.item())              #
        self.sen_sqr_list.append(self.epsi_sp**2)       #
        self.term_volt = V_term.item()                  #

        self.remaining_time -= self.dt                  #

        done = bool(self.time_horizon_counter >= self.training_duration
                    or np.isnan(V_term))

        # done = bool(self.time_horizon_counter >= self.training_duration
        #             or np.isnan(V_term)
        #             or done_flag is True)

        if not done:
            reward = self.reward_function(self.epsi_sp.item())

        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = self.reward_function(self.epsi_sp.item())

        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                  "You are calling 'step()' even though this "
                  "environment has already returned done = True. You "
                  "should always call 'reset()' once you receive 'done = "
                  "True' -- any further steps are undefined behavior.")

            self.steps_beyond_done += 1
            reward = 0.0

        # Log Tensorboard Variables
        self.tb_C_se0 = theta[0].item()
        self.tb_C_se1 = theta[1].item()
        self.tb_epsi_sp = self.epsi_sp
        self.tb_state_of_charge = soc_new[1].item()
        self.tb_state_of_charge_1 = soc_new[0].item()
        # self.tb_dCs_deps = self.state[0]

        self.tb_term_volt = self.term_volt
        self.tb_input_current = input_current

        self.tb_instantaneous_reward = reward
        self.tb_reward_mean = self.increment_mean(reward, self.tb_reward_mean, self.tb_reward_mean_counter)
        self.tb_reward_mean_counter += 1
        self.tb_reward_sum += reward

        if self.log_state is True:

            self.writer.add_scalar('Battery/C_se0', self.tb_C_se0, self.global_counter)
            # self.writer.add_scalar('Battery/C_se1', self.tb_C_se1, self.global_counter)
            self.writer.add_scalar('Battery/Epsi_sp', self.tb_epsi_sp, self.global_counter)
            self.writer.add_scalar('Battery/SOC', self.tb_state_of_charge, self.global_counter)
            # self.writer.add_scalar('Battery/SOC_1', self.tb_state_of_charge_1, self.global_counter)
            self.writer.add_scalar('Battery/dCse_dEpsilon', self.tb_dCs_deps, self.global_counter)
            self.writer.add_scalar('Battery/Remaining_Time', self.remaining_time, self.global_counter)


            self.writer.add_scalar('Battery/Term_Voltage', self.tb_term_volt, self.global_counter)
            self.writer.add_scalar('Battery/Input_Current', self.tb_input_current, self.global_counter)
            self.writer.add_scalar('Battery/Instant Reward', self.tb_instantaneous_reward, self.global_counter)

            self.writer.add_scalar('Battery/Cum. Reward', self.tb_reward_sum, self.global_counter)
            self.writer.add_scalar('Battery/Avg. Reward', self.tb_reward_mean, self.global_counter)
            self.writer.add_scalar('Battery/Num. Episodes', self.episode_counter, self.global_counter)

            # if self.global_counter == 50000 or self.global_counter % 1000000 == 0:
            #     import time
            #     current_time = time.ctime(time.time())
            #
            #     logging.info(f"Current TimeStep: {self.global_counter},  Start Time: {self.start_time}, Current Time: {current_time}")

        if self.global_counter == 50000 or self.global_counter % 1000000 == 0:
            import time
            current_time = time.ctime(time.time())

            logging.info(
                f"Current TimeStep: {self.global_counter},  Start Time: {self.start_time}, Current Time: {current_time}")

        self.rec_epsi_sp.append(self.tb_epsi_sp.item())
        self.rec_input_current.append(self.tb_input_current)
        self.rec_state_of_charge.append(self.tb_state_of_charge)
        self.rec_term_volt.append(self.tb_term_volt)
        self.rec_time.append(self.time)

        self.time += self.time_step

        self.time_horizon_counter += 1
        self.global_counter += 1
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.step_counter = 0
        self.time_horizon_counter = 0
        self.episode_counter += 1

        self. sen_sqr_list = []
        self.sen_list = []
        self.remaining_time = self.training_duration

        self.state_of_charge = self.SOC_0

        new_states, outputs, sensitivity, soc_new, theta, V_term = self.model.SPMe_step(states=self.sim_states, I_input=0.0, init_Flag=True)

        # self.state = [self.SOC_0]

        self.sim_states = new_states
        self.state = [soc_new[1].item()]
        self.steps_beyond_done = None
        return np.array(self.state)

    def plot(self):

        print(self.sen_list)
        plt.plot(self.sen_list)
        plt.show()


if __name__ == '__main__':

    env = SPMenv()
    env.reset()
    soc_list = []

    for time in range(1800):

        input = env.action_space.sample()

        input = np.array(input)

        obs, reward, done, _ = env.step(input)

        # obs, reward, done, _ = env.step(np.array([0]))

        soc_list.append(obs.item())

    # env.plot()

    plt.plot(soc_list)
    plt.show()
