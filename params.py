import numpy as np
L = 1
AREA_SIZE = L**3
SPEED = 50
MAX_MOVE = 0.5*L
#SIGMA = (SPEED * 1000. / 24. / AREA_SIZE) / (3. * np.sqrt(2))
UNSEEN = 0
SEEN = 1
#SOCIAL_DISTANCE = 0.0000007
SOCIAL_DISTANCE = L/10
BETA1 = 1  # amylase rate
BETA2 = 1  # glucose to lactate rate
BETA3 = 0.3  # lactate to electron rate

FRAME_RATE = 1  # Refresh graphics very FRAME_RATE hours
# DENSITY = 100
starch_density = 100
ecoli_density = 100
so_density = 100
e_density = 0
l_density = 0
g_density = 0
duration = 1
