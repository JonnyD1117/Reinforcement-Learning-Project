# Add Imports
from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity as SPMe
from Large_Discrete_SPMe_w_remaining_time_env import SPMenv as Discrete_SPMe_env

import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import logging
import time

########################################################################################################################

########################################################################################################################

def Discretize_Value(input_val, input_range, num_disc):








# Initialize SPMe Model
SOC_0 = .5
sim_state = SPMe.full_init_state
SPMe.__init__(init_soc=SOC_0)

"""
SPMe Battery is +- 1C-rate capped

STATE(S): Battery SOC or Concentration or dCe_dEps_sp
ACTION: Input Current

Discretization: 
    S: 
    A: 
"""


# Initialize Q-Learning Parameters:

num_avg_runs = 1
num_episodes = 1
episode_duration = 1800

# Initialize Q-Learning Table

num_q_states = 10
num_q_actions = 10

Q = np.zeros(num_q_states, num_q_actions)


for avg_num in range(num_avg_runs):
    for eps in range(num_episodes):
        for step in range(episode_duration):



            [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done] = SPMe.SPMe_step(full_sim=True, states=sim_state, I_input=0)

            sim_state = [bat_states, new_sen_states]
            state = SPMe.unpack_states(bat_states, new_sen_states, outputs, sensitivity_outputs)
            # state.append(remaining_time)





# Initialize States
remaining_time =





























# Compute New Battery & Sensivitiy States
[bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done_flag] \
    = SPMe.SPMe_step(full_sim=True, states=sim_state_before, I_input=input_current)

soc_list.append(soc_new[1].item())

# Unpack System, Simulation, and Sensitivity States and Outputs
sim_state_after = [bat_states, new_sen_states]
sim_state_before = sim_state_after
state_of_charge = soc_new[1].item()
state_output = outputs
state = unpack_states(bat_states, new_sen_states, outputs, sensitivity_outputs)
state.append(remaining_time)

# Set Key System Variables
epsi_sp = sensitivity_outputs['dV_dEpsi_sp']

sen_list.append(epsi_sp)
sen_sqr_list.append(epsi_sp ** 2)
term_volt = V_term.item()

# Compute Termination Conditions
concentration_pos = state_output['yp']
concentration_neg = state_output['yn']

remaining_time -= dt

# Discretize State-Space
# Discretize Action-Space

# Initialize Q-Learning Params...etc


# Loop over Averaging X times

# Loop over Y Episodes times

# Loop over T Episodes Time-Steps