from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity
import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import logging
import time




def increment_mean(new_value, prev_mean, mean_counter):

    if mean_counter == 0:
        new_mean = prev_mean

    else:
        new_mean = prev_mean + ((new_value-prev_mean)/mean_counter)

    return new_mean


class SPMenv(gym.Env):

    def __init__(self, time_step=1, training_duration=1800, log_dir=None, log_trial_name=None,log_data=True, SOC=.5):
        # super(SingleParticleModelElectrolyte_w_Sensitivity).__init__()

        self.global_counter = 0
        self.episode_counter = 0
        self.time_horizon_counter = 0
        self.training_duration = training_duration
        self.remaining_time = self.training_duration

        self.log_state = log_data

        if self.log_state is True:
            if log_dir is None:
                print("NO Logging Directory Name Given")
                exit()

            elif log_trial_name is None:
                print("NO Logging Trial Name Given")
                exit()

            else:
                self.writer = SummaryWriter(f'./{log_dir}/log_files/{log_trial_name}')
                logging.basicConfig(filename=f'./{log_dir}/log_files/{log_trial_name}/{log_trial_name}.log', level=logging.INFO)

        self.soc_list = []

        self.cs_max_n = (3.6e3 * 372 * 1800) / 96487
        self.cs_max_p = (3.6e3 * 274 * 5010) / 96487

        self.max_sen = 0
        self.time_step = time_step
        self.start_time = time.ctime(time.time())
        self.dt = 1
        self.step_counter = 0
        self.SPMe = SingleParticleModelElectrolyte_w_Sensitivity(timestep=self.time_step, init_soc=SOC)

        upper_state_limits = np.array([np.inf, np.inf], dtype=np.float32)
        lower_state_limits = np.array([-np.inf, 0], dtype=np.float32)
        max_C_val = np.array([25.67*1], dtype=np.float32)

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
        action_list = np.arange(-3, 3, 1)
        self.action_space = spaces.Discrete(len(action_list))
        self.action_dict = {index: value*max_C_val for index, value in enumerate(action_list)}

        self.observation_space = spaces.Box(lower_state_limits, upper_state_limits, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.sim_state = None
        self.sim_state_before = None
        self.sim_state_after = None

        self.steps_beyond_done = None
        self.state_output = None

        # TensorBoard Variables
        self.tb_C_se0 = None
        self.tb_C_se1 = None
        self.tb_epsi_sp = None
        # self.tb_dCse_dEpsi = None
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

    @staticmethod
    def unpack_states(bat_states, sen_states, state_out, sen_out):

        x1 = bat_states['xn']
        x2 = bat_states['xp']
        x3 = bat_states['xe']

        x4 = sen_states['Sepsi_p']
        x5 = sen_states['Sepsi_n']
        x6 = sen_states['Sdsp_p']
        x7 = sen_states['Sdsn_n']

        yn = state_out["yn"]
        yp = state_out["yp"]
        yep = state_out["yep"]


        # print(f"x1: {x1}, x2: {x2}, x3: {x3}, x4: {x4}, x5: {x5}, x6: {x6}, x7: {x7}, yn: {yn}, yp: {yp}, yep:{yep}")
        # print(f"x1: {len(x1)}, x2: {len(x2)}, x3: {len(x3)}, x4: {len(x4)}, x5: {len(x5)}, x6: {len(x6)}, x7: {len(x7)}, yn: {len(yn)}, yp: {len(yp)}, yep:{len(yep)}")


        dV_dDsn = sen_out["dV_dDsn"]
        dV_dDsp = sen_out["dV_dDsp"]
        dCse_dDsn = sen_out["dCse_dDsn"]
        dCse_dDsp = sen_out["dCse_dDsp"]
        dV_dEpsi_sn = sen_out["dV_dEpsi_sn"]
        dV_dEpsi_sp = sen_out["dV_dEpsi_sp"]
        dCse_dEpsi_sp = sen_out["dCse_dEpsi_sp"]
        dCse_dEpsi_sn = sen_out["dCse_dEpsi_sn"]

        return [dCse_dEpsi_sp.item()]

    def get_time(self):
        total_time = self.time_step*self.time_horizon_counter
        return total_time

    def reward_function(self, sensitivity_value, action, Cse_val, Cse_threshold=.8):

        reward = sensitivity_value**2

        # reward = sum(self.sen_sqr_list)
        return reward

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        # print(f"ACTION: {action}")
        action = action.item()

        input_current = self.action_dict[action]

        if self.step_counter == 0:
            self.sim_state_before = self.SPMe.full_init_state
            input_current = np.array(0)
            self.step_counter += 1
        else:
            self.step_counter += 1

        # Compute New Battery & Sensivitiy States
        [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done_flag] \
            = self.SPMe.SPMe_step(full_sim=True, states=self.sim_state_before, I_input=input_current)

        # If Terminal Voltage Limits are hit, maintain the current state
        # if V_term > self.max_term_voltage or V_term < self.min_term_voltage:
        #
        #     [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse,
        #      done] = self.SPMe.SPMe_step(full_sim=True, states=self.sim_state_before, I_input=0)

        self.soc_list.append(soc_new[1].item())

        # Unpack System, Simulation, and Sensitivity States and Outputs
        self.sim_state_after = [bat_states, new_sen_states]
        self.sim_state_before = self.sim_state_after
        self.state_of_charge = soc_new[1].item()
        self.state_output = outputs
        self.state = self.unpack_states(bat_states, new_sen_states, outputs, sensitivity_outputs)
        self.state.append(self.remaining_time)

        # Set Key System Variables
        self.epsi_sp = sensitivity_outputs['dV_dEpsi_sp']

        self.sen_list.append(self.epsi_sp)
        self.sen_sqr_list.append(self.epsi_sp**2)
        self.term_volt = V_term.item()

        # Compute Termination Conditions
        concentration_pos = self.state_output['yp'] 
        concentration_neg = self.state_output['yn']

        self.remaining_time -= self.dt


        done = bool(self.time_horizon_counter >= self.training_duration
                    or np.isnan(V_term)
                    or done_flag is True)

        if not done:
            reward = self.reward_function(self.epsi_sp.item(), action, soc_new[0].item(), Cse_threshold=.8)

        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = self.reward_function(self.epsi_sp.item(), action, soc_new[0].item(), Cse_threshold=.8)

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
        self.tb_dCs_deps = self.state[0]

        self.tb_term_volt = self.term_volt
        self.tb_input_current = input_current

        self.tb_instantaneous_reward = reward
        self.tb_reward_mean = increment_mean(reward, self.tb_reward_mean, self.tb_reward_mean_counter)
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

            if self.global_counter == 50000 or self.global_counter % 1000000 == 0:
                import time
                current_time = time.ctime(time.time())

                logging.info(f"Current TimeStep: {self.global_counter},  Start Time: {self.start_time}, Current Time: {current_time}")

        self.rec_epsi_sp.append(self.tb_epsi_sp.item())
        self.rec_input_current.append(self.tb_input_current)
        self.rec_state_of_charge.append(self.tb_state_of_charge)
        self.rec_term_volt.append(self.tb_term_volt)
        self.rec_time.append(self.time)

        self.time += self.time_step

        self.time_horizon_counter += 1
        self.global_counter += 1
        return np.array(self.state), reward, done, {}

    def reset(self, test_flag=False):
        self.step_counter = 0
        self.time_horizon_counter = 0
        self.episode_counter += 1
        # print("RESET CALLED")

        self. sen_sqr_list = []
        self.sen_list = []

        self.remaining_time = self.training_duration

        self.state = None


        # state_noise = np.random.normal(loc=.125, scale=1)

        if test_flag is True:
            self.state_of_charge = self.SOC_0
            self.SPMe.__init__(init_soc=self.SOC_0)

        else:

            self.state_of_charge = self.SOC_0
            self.SPMe.__init__(init_soc=self.SOC_0)

            # self.state_of_charge = (np.random.uniform(.65, .99, 1)).item()
            # self.SPMe.__init__(init_soc=self.state_of_charge)

        # self.state_of_charge = self.SOC_0
        # self.SPMe.__init__(init_soc=self.SOC_0)

        self.sim_state = self.SPMe.full_init_state

        [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done] = self.SPMe.SPMe_step(
            full_sim=True, states=self.sim_state, I_input=0)

        self.sim_state = [bat_states, new_sen_states]
        self.state = self.unpack_states(bat_states, new_sen_states, outputs, sensitivity_outputs)
        self.state.append(self.remaining_time)

        self.steps_beyond_done = None
        return np.array(self.state)


if __name__ == '__main__':

    gym = SPMenv()

    gym.reset()
