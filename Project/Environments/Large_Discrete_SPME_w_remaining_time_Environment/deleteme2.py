import numpy as np



max_val = 25.7



thing  = np.arange(-1,1, .1)

action_dict = {index: value * max_val for index, value in enumerate(thing)}

print(action_dict)

