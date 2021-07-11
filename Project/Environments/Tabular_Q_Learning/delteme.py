import numpy as np




ep_dec = np.arange(.5, .01, -.001)

print(ep_dec)



# def Discretize_Value(input_val, input_range, num_disc, zipped=False):
#     """
#     Uniform Discretization of input variable given the min/max range of the input variable and the total number of discretizations desired
#     """
#     step_size = (input_range[1] - input_range[0]) / num_disc                    # Compute the Average Step-Size for "num_disc" levels
#     discrete_values = np.arange(input_range[0], input_range[1], step_size)      #
#     index_values = np.arange(0, num_disc, 1)
#     zipped_var = zip(index_values, discrete_values)
#
#     if zipped is False:
#
#         for i in index_values:
#
#             if discrete_values[i] <= input_val <= discrete_values[i+1]:
#                 # print(f"Upper Bound: {discrete_values[i+1]}")
#                 # print(f"Lower Bound: {discrete_values[i]}")
#                 # print(f"Input Value: {input_val}")
#
#                 upper_error = np.abs(discrete_values[i+1] - input_val)
#                 lower_error = np.abs(discrete_values[i] - input_val)
#
#                 # print(f"Upper Bound Error: {upper_error}")
#                 # print(f"Lower Bound Error: {lower_error}")
#
#                 if lower_error > upper_error:
#                     output_val = discrete_values[i+1]
#                     output_index = (i + 1)
#                     break
#                     # print(f"Output Value: {output_val}")
#
#                 else:
#                     output_val = discrete_values[i]
#                     output_index = i
#                     break
#                     # print(f"Output Value: {output_val}")
#         return output_val, output_index
#
#     else:
#         return zipped_var
#
#
#
# #
# # val, ind = Discretize_Value(12.21, [-25.7,25.7], 100)
# #
# # print(val)
# # print(ind)
#
# Q = np.zeros(shape=[2,2])
# Q[1][0] = 1
#
# max_Q = np.max(Q[0][:])
#
# print(Q)
# print(f"Max Q = {max_Q}")


#
# ind = np.unravel_index(np.argmax(Q, axis=None), Q.shape)
# print(ind)
#
# action_ind = ind[1]
#
# print(action_ind)


# Initialize Q-Learning Table

# num_q_states = 10
# num_q_actions = 10
#
# max_state_val = 1
# min_state_val = 0
#
# max_action_val = 25.7
# min_action_val = -25.7
#
# zipped_actions = Discretize_Value(input_val=None, input_range=[min_action_val, max_action_val], num_disc=num_q_actions, zipped=True)
# thing1, thing2 = Discretize_Value(input_val=10, input_range=[min_action_val, max_action_val], num_disc=num_q_actions, zipped=False)
#
# action_list = {ind:value for ind, value in zipped_actions}
#
# print(action_list)