import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from SPMe_Baseline_Params import SPMe_Baseline_Parameters
from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity
from time import sleep

input_profile_1C_pulses = loadmat("./Input_Profiles/comparisonCurrent_1C_Pulse.mat")
input_profile_2C_CC = loadmat("./Input_Profiles/comparisonCurrent_2C_CC.mat")
input_profile_FUDs = loadmat("./Input_Profiles/comparisonCurrent_FUDS.mat")



current_data_1C_pulse = input_profile_1C_pulses["currentData"][0, 0][0][0][:]
time_data_1C_pulse = input_profile_1C_pulses["currentData"][0, 0][1][0][:]

current_data_2C_CC = input_profile_2C_CC["currentData"][0, 0][0][0][:]
time_data_2C_CC = input_profile_2C_CC["currentData"][0, 0][1][0][:]

current_data_FUDs = input_profile_FUDs["currentData"][0, 0][0][0][:]
time_data_FUDs = input_profile_FUDs["currentData"][0, 0][1][0][:]

sim_time_1C_pulses = time_data_1C_pulse[-1]
sim_time_2C_cc = time_data_2C_CC[-1]
sim_time_FUDs = time_data_FUDs[-1]

dt_1C_pulse = time_data_1C_pulse[2] - time_data_1C_pulse[1]
dt_2C_cc = time_data_2C_CC[2] - time_data_2C_CC[1]
dt_FUDs = time_data_FUDs[2] - time_data_FUDs[1]

# print(f"Pulses Sim. Time {sim_time_1C_pulses}")
# print(f"2CC Sim. Time {sim_time_2C_cc}")
# print(f"FUDs Sim. Time {sim_time_FUDs}")
#
# print(f"Pulses dt: {dt_1C_pulse}")
# print(f"2CC dt: {dt_2C_cc}")
# print(f"FUDs dt: {dt_FUDs}")
#
# plt.figure()
# plt.plot(current_data_1C_pulse)
#
# plt.figure()
# plt.plot(current_data_2C_CC)
#
# plt.figure()
# plt.plot(current_data_FUDs)
# plt.show()

input_list = [(current_data_1C_pulse, sim_time_1C_pulses, dt_1C_pulse), (current_data_2C_CC, sim_time_2C_cc, dt_2C_cc), (current_data_FUDs, sim_time_FUDs,dt_FUDs ) ]
input_tuple = input_list[1]

input_signal = input_tuple[0]
sim_time = input_tuple[1]
time_step = input_tuple[2]


soc_list = []
SPMe = SingleParticleModelElectrolyte_w_Sensitivity(sim_time=sim_time, init_soc=.5, timestep=time_step)

[xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term, time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp] \
    = SPMe.sim(CC=False, zero_init_I=False, I_input=input_signal, plot_results=True)


# print(f" Minimum SOC={np.min(soc)} : Maximum SOC={np.max(soc)}")
# print(f"Electrode #1  Concentration Minimum={np.min(theta_n)} : Maximum={np.max(theta_n)}")
# print(f"Electrode #2  Concentration Minimum={np.min(theta_p)} : Maximum={np.max(theta_p)}")