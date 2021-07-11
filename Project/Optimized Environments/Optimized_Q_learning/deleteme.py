
import numpy as np

num_disc = 10

input_range = [-25.7, 25.7]
max_state_val = 1
min_state_val = 0
num_q_states = 10

# print(np.random.uniform(0, 1, size=10))

print(np.ones(shape=(2,3)))

# def Discretization_Dict(input_range, num_disc):
#
#     step_size = (input_range[1] - input_range[0]) / num_disc
#     discrete_values = np.arange(input_range[0], input_range[1], step_size)
#     index_values = np.arange(1, (num_disc + 1), 1)
#     zipped_var = {i: discrete_values[i] for i in range(len(discrete_values))}
#
#     return [discrete_values, index_values, zipped_var]
#
#
# [soc_state_values, soc_state__index, soc_state_dict] = Discretization_Dict([min_state_val, max_state_val], num_q_states)
#
#
# print(soc_state_values)
#
# print(np.shape(soc_state_values))