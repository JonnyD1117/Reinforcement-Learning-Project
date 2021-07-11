import numpy as np


def Discretize_Value(input_val, input_range, num_disc, zipped=False):
    """
    Uniform Discretization of input variable given the min/max range of the input variable and the total number of discretizations desired
    """
    step_size = (input_range[1] - input_range[0]) / num_disc                    # Compute the Average Step-Size for "num_disc" levels
    discrete_values = np.arange(input_range[0], input_range[1], step_size)      #
    index_values = np.arange(0, num_disc, 1)
    zipped_var = zip(index_values, discrete_values)

    if zipped is False:

        for i in index_values:

            if input_val < discrete_values[0]:

                output_val = discrete_values[0]
                output_index = 0

                return output_val, output_index

            elif input_val > discrete_values[-1]:

                output_val = discrete_values[-1]
                output_index = num_disc - 1

                return output_val, output_index

            else:

                if discrete_values[i] <= input_val < discrete_values[i+1]:

                    upper_error = np.abs(discrete_values[i+1] - input_val)
                    lower_error = np.abs(discrete_values[i] - input_val)

                    if lower_error > upper_error:
                        output_val = discrete_values[i+1]
                        output_index = (i + 1)

                    else:
                        output_val = discrete_values[i]
                        output_index = i

                    return output_val, output_index

    else:
        return zipped_var


def epsilon_greedy_policy(state_ind, Q_table, Q_dims, epsilon):
    num_act = Q_dims[1]
    num_states = Q_dims[0]

    if np.random.uniform(0, 1) <= epsilon:
        action_ind = np.random.randint(0, num_act)

    else:
        ind = np.unravel_index(np.argmax(Q_table[state_ind][:], axis=None), Q_table.shape)
        action_ind = ind[1]

    return action_ind



Q_tab = np.zeros([10,10])
Q_tab[0][5] = 1


# print(np.unravel_index(np.argmax(Q_tab[0][:], axis=None), Q_tab.shape))
#
# print(Q_tab)