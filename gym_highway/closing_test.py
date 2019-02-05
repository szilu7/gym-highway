import gym
import gym_highway
import numpy as np
import math
# import time
import matplotlib.pyplot as plt

from gym_highway.modell.environment_vehicle import Envvehicle

env = gym.make('EPHighWay-v0')
env.reset()

action=12


envdict=env.unwrapped.envdict
modell=env.unwrapped.modell


for i in range(envdict['lane_count']):
    veh=Envvehicle(envdict)
    veh.y=0
    veh.vx=36
    modell.lanes[0]=[veh]
    veh=Envvehicle(envdict)
    veh.y=8
    veh.vx = 36
    modell.lanes[2]=[veh]
    veh=Envvehicle(envdict)
    veh.y=4
    veh.x=-10
    veh.vx = 36
    modell.lanes[1]=[veh]
    veh=Envvehicle(envdict)
    veh.y=5.5
    veh.vx = 36
    modell.lanes[1].append(veh)
    veh=Envvehicle(envdict)
    veh.y=4
    veh.x = 10
    veh.vx = 36
    modell.lanes[1].append(veh)





modell.searchEgoVehicle(1)
env.unwrapped.state=modell.generate_state_for_ego()
rewards=env.unwrapped.calcreward()
print(rewards)
modell.render(True, env.unwrapped.rewards)
plt.show(True)



