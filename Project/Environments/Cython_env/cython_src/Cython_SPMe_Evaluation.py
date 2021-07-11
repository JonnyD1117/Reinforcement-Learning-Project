# from Cythonized_SPMe_w_Sensitivity_Params import profile
import time

from Cythonized_SPMe_STEP import SPMe_step as cython_step
from Python_SPMe_STEP import SPMe_step as python_step
# from Opt_Python_SPMe_STEP import SPMe_step as cython_step

import numpy as np


new_states = None
soc_list = []

mean_cython_time = []

for _ in range(2000):
    cython_init_time = time.time_ns()
    initial_step = True
    [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done_flag] = cython_step(states=new_states, I_input=-25.7, full_sim=False, init_flag=initial_step)
    cython_final_time = time.time_ns()
    mean_cython_time.append((cython_final_time-cython_init_time)/1e9)


print(f"Cython Mean Step Time: {np.mean(mean_cython_time)} seconds")



# cython_init_time = time.time_ns()
# for num_eps in range(1):
#     for t in range(1800):
#         if t == 0:
#             initial_step = True
#         else:
#             initial_step = False
#
#         [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse,
#              done_flag] = cython_step(states=new_states, I_input=-25.7, full_sim=False, init_flag=initial_step)
#
#         soc_list.append(soc_new[0].item())
#
#         new_states = [bat_states, new_sen_states]
#
# cython_final_time = time.time_ns()




new_states = None
soc_list = []

mean_python_time = []

for _ in range(2000):
    python_init_time = time.time_ns()
    initial_step = True
    [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done_flag] = python_step(states=new_states, I_input=-25.7, full_sim=False, init_flag=initial_step)
    python_final_time = time.time_ns()
    mean_python_time.append((python_final_time - python_init_time) / 1e9)

print(f"Python Mean Step Time: {np.mean(mean_python_time)} seconds")


print(f"Average Delta: {np.mean(mean_python_time) - np.mean(mean_cython_time)}")



# python_init_time = time.time_ns()
# for num_eps in range(1):
#     for t in range(1800):
#         if t == 0:
#             initial_step = True
#         else:
#             initial_step = False
#
#         [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse,
#              done_flag] = python_step(states=new_states, I_input=-25.7, full_sim=False, init_flag=initial_step)
#
#         soc_list.append(soc_new[0].item())
#
#         new_states = [bat_states, new_sen_states]
#
# python_final_time = time.time_ns()

#
# print(f"Python Episode Time: {(python_final_time - python_init_time)/1e9} seconds")
