import timeit
from Discrete_SPMe_w_remaining_time_env import SPMenv
import numpy as np
import cProfile




# print(timeit.timeit(setup="""from Discrete_SPMe_w_remaining_time_env import SPMenv
# import numpy as np""",
#                     stmt='SPMenv(log_data=False).step(np.array([1]))',
#                     number=1800))


cProfile.run('SPMenv(log_data=False).step(np.array([1]))')