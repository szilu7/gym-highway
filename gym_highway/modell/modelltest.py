import math

import matplotlib.pyplot as plt
import numpy as np
#  from gym_highway.modell.modell import Modell
from modell import Modell

envdict = {'length_forward': 1000, 'length_backward': 500, 'dt': 0.2}
#envdict = {'length_forward': 400, 'length_backward': 400, 'dt': 0.2}
envdict['lane_width'] = 4

envdict['lane_count'] = 3
# Vehicle Generation Parameters
envdict['density_lanes'] = 10  # 8 #[vehicle density vehicle/km]

envdict['speed_mean_lane0'] = 110.0 / 3.6  # generated vehicle desired speed mean [m/s]
envdict['speed_std_lane0'] = 10.0 / 3.6    # generated vehicle desired speed deviation [m/s]
envdict['speed_mean_lane1'] = 140.0 / 3.6  # generated vehicle desired speed mean [m/s]
envdict['speed_std_lane1'] = 10.0 / 3.6    # generated vehicle desired speed deviation [m/s]
envdict['speed_mean_lane2'] = 160.0 / 3.6  # generated vehicle desired speed mean [m/s]
envdict['speed_std_lane2'] = 10.0 / 3.6    # generated vehicle desired speed deviation [m/s]
envdict['speed_mean_lane3'] = 170.0 / 3.6  # generated vehicle desired speed mean [m/s]
envdict['speed_std_lane3'] = 10.0 / 3.6    # generated vehicle desired speed deviation [m/s]
envdict['speed_mean_lane4'] = 180.0 / 3.6  # generated vehicle desired speed mean [m/s]
envdict['speed_std_lane4'] = 10.0 / 3.6    # generated vehicle desired speed deviation [m/s]

envdict['speed_ego_desired'] = 100.0 / 3.6  # generated vehicle desired speed deviation [m/s]


mod=Modell(envdict)
mod.warmup(False)
mod.searchEgoVehicle(preferredlaneid=0)
for i in range(50000):
    #steer=0.001*math.sin(i/100)
    steer = 0
    action=np.array([steer,0.301]) # [steering, acceleration]
    success,cause= mod.onestep(action)
    if not success:
        print(cause)
        break
    mod.render(True)
