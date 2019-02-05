import numpy as np
import matplotlib.pyplot as plt
import gym_highway.modell.environment_vehicle as ev
from itertools import cycle
cycol = cycle('bgrcmk')
envdict ={'length_forward' :2000 ,'length_backward' :1000 ,'dt' :0.2}

vehicles=[ev.Envvehicle(envdict),None]
nextvehicle=30

for j in range(10000):
    plt.hold(False)
    if vehicles[0].x>-envdict['length_backward']+nextvehicle:
        vehicles.insert(0,ev.Envvehicle(envdict))
        vehicles[0].vx=vehicles[1].vx
        vehicles[0].color=next(cycol)
        nextvehicle=30+max(10,150+np.random.randn()*10)
    for i in range(len(vehicles)-1):

        if vehicles[i] is not None:
            vehicles[i].step(vehicles[i+1],None,None)
            vehicles[i].render()
            plt.hold(True)

    plt.axis('equal')
    plt.xlim([-envdict['length_backward'], envdict['length_forward']])
    plt.show(False)

    plt.pause(0.001)

