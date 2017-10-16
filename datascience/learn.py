import numpy as np

world_alcohol = np.genfromtxt('world_alcohol.csv',dtype="U75", delimiter=",", skip_header = 1)
world_alcohol_dtype = world_alcohol.dtype
print(world_alcohol)