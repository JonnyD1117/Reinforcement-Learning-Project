import gym
from gym import error, spaces, utils, logger
from torch.utils.tensorboard import SummaryWriter

import random

import numpy as np


class DiscreteSimpleSOC(gym.Env):

    def __init__(self, log_state=True):

        # self.cap = 2500 # mAh
        self.cap = 25.670*3600  # Ah*3600 := Amp*seconds
        self.dt = 1  # 1 second
        self.c_rate = 1

        self.training_duration = 1800
        # self.training_duration = 3600

        self.time_horizon_counter = 0
        self.global_counter = 0
        self.episode_counter = 0
        self.log_state = log_state

        if self.log_state is True:
            self.writer = SummaryWriter('./log_files/tb_log/Round1/Disc_T15')

        state_limits = np.array([np.inf], dtype=np.float32)
        action_limits = np.array([25.67*self.c_rate], dtype=np.float32)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-state_limits, state_limits, dtype=np.float32)

        self.action_dict = {0:-1, 1:0, 2: 1}
        self.state = None
        self.SOC_init = .5
        self.SOC_0 = self.SOC_init
        self.SOC = 0

        # TensorBoard Logging
        self.tb_input_current = None
        self.tb_state_of_charge = self.SOC_0
        self.tb_reward_list = []
        self.tb_reward_mean = None
        self.tb_reward_sum = None
        self.tb_instantaneous_reward = None


    def step(self, action):

        input_current = 25.67*self.action_dict[action.item()]

        # SOC Integrator Dynamics (NOTE: Negative Input Current -> Charging)
        self.SOC = self.SOC_0 - (1./self.cap)*input_current*self.dt
        self.SOC_0 = self.SOC

        # Max and Min SOC Constraints
        ##############################################
        ## Testing Penalty Based SOC limiting
        ##############################################
        # if self.SOC > 1.0:
        #     self.SOC = 1.
        #
        # elif self.SOC < 0.0:
        #     self.SOC = 0.
        #
        # else:
        #     self.SOC = self.SOC
        ###############################################
        # Determine if the Episode has Reached it's termination Time
        done = bool(self.time_horizon_counter >= self.training_duration)

        if not done:
            reward = self.reward_function(self.SOC)

        else:

            logger.warn(
                  "You are calling 'step()' even though this "
                  "environment has already returned done = True. You "
                  "should always call 'reset()' once you receive 'done = "
                  "True' -- any further steps are undefined behavior.")

            reward = 0.0

        # Log TensorBoard Variables
        self.tb_input_current = input_current
        self.tb_state_of_charge = self.SOC
        self.tb_reward_list.append(reward)
        self.tb_reward_mean = np.mean(self.tb_reward_list)
        self.tb_reward_sum = np.sum(self.tb_reward_list)

        self.tb_instantaneous_reward = reward

        if self.log_state is True:
            self.writer.add_scalar('SimpleSOC/SOC', self.tb_state_of_charge, self.global_counter)
            self.writer.add_scalar('SimpleSOC/Input_Current', self.tb_input_current, self.global_counter)
            self.writer.add_scalar('SimpleSOC/Instant Reward', self.tb_instantaneous_reward, self.global_counter)
            self.writer.add_scalar('SimpleSOC/Cum. Reward', self.tb_reward_sum, self.global_counter)
            self.writer.add_scalar('SimpleSOC/Avg. Reward', self.tb_reward_mean, self.global_counter)
            self.writer.add_scalar('SimpleSOC/Num. Episodes', self.episode_counter, self.global_counter)

        self.state = [self.SOC]
        self.time_horizon_counter += 1
        self.global_counter += 1

        return np.array(self.state), reward, done, {}

    def reset(self):

        self.episode_counter += 1
        self.time_horizon_counter = 0
        self.SOC = 0.
        self.SOC_0 = self.SOC_init
        # self.SOC_0 = random.uniform(1.2,-.2)
        # self.SOC_0 = random.uniform(1,0)

        # self.state = np.array([random.uniform(1,0)])

        self.state = [self.SOC_0]

        return np.array(self.state)


    def reward_function(self, reward_input):

        min_reward_thres = .60
        max_reward_thres = .85

        # penalty =0
        #
        # reward1 = 1
        if min_reward_thres <= reward_input <= max_reward_thres:

            reward1 = 1

        else:
            reward1 = -1

        # if reward_input > 1 or reward_input < 0:
        #     penalty = -5
        #
        # reward = reward1 + penalty

        return reward1