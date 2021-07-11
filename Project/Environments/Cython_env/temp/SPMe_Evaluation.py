# from Cythonized_SPMe_w_Sensitivity_Params import profile
import time

# from Cythonized_SPMe_STEP import SPMe_step as cython_step
from Python_SPMe_STEP import SPMe_step as python_step
# from Opt_Python_SPMe_STEP import SPMe_step as cython_step

import numpy as np


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