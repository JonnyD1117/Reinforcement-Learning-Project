from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity as SPMe
import time
from tqdm import tqdm
import numpy as np

SOC_0 = .5
model = SPMe(init_soc=SOC_0)
sim_state = model.full_init_state


step_times = []

for eps in tqdm(range(500)):
    for t_sep in range(1800):
        init_time = time.time_ns()

        [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done] = model.SPMe_step(
            full_sim=True, states=sim_state, I_input=0)

        f_time = time.time_ns()


        total_run_time = (f_time - init_time)/1000000000

        step_times.append(total_run_time)

print(f"Mean Time for STEP: {np.mean(step_times)}")