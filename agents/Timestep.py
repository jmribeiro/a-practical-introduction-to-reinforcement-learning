from collections import namedtuple

# This one does not require a class
Timestep = namedtuple("Timestep", "observation action reward next_observation terminal info")
