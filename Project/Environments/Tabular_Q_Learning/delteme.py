import numpy as np


def Discretize_Value(input_val, input_range, num_disc):
    """
    Uniform Discretization of input variable given the min/max range of the input variable and the total number of discretizations desired
    """

    step_size = (input_range[1] - input_range[0]) / num_disc                    # Compute the Average Step-Size for "num_disc" levels

    discrete_values = np.arange(input_range[0], input_range[1], step_size)      #
    index_values = np.arange(0, num_disc, 1)
    zipped_var = zip(index_values, discrete_values)

    return index_values, discrete_values