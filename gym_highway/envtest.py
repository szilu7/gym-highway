import gym
import gym_highway
import numpy as np
import math
import time
import sys
import matplotlib.pyplot as plt
from msvcrt import getch, kbhit
from msvcrt import getch, kbhit
env = gym.make('EPHighWay-v0')
env.reset()

global keyaction
keyaction='0'

def press(event):
    global keyaction
    keyaction=event.key
    sys.stdout.flush()

def contaction(x):
    return {
        '4': np.array([0.003,0.]),
        '6': np.array([-0.003,0.]),
        '8': np.array([0.000,1.]),
        '2': np.array([0.000,-1.]),
        '0': np.array([0.000,0.]),
    }.get(x,np.array([0.000,0.]))

def discaction(x):
    return {
        '4': 22,
        '6': 2,
        '8': 13,
        '2': 11,
        '0': 12,
    }.get(x,12)

def smaction(x):
    return {
        '4': 0,
        '6': 1
    }.get(x,2)


st = [-0.003, -0.0005, 0, 0.0005, 0.003]
ac = [-6.0, -2.0, 0.0, 2.0, 3.5]

action=12
t = time.time()

for i in range(10000):
    env.render()
    plt.gcf().canvas.mpl_connect('key_press_event', press)

    if env.unwrapped.envdict['action_space'] == 'CONTINUOUS':
        action = contaction(keyaction)
        keyaction='0'
    elif env.unwrapped.envdict['action_space'] == 'DISCRETE':
        action = discaction(keyaction)
        keyaction = '0'
    elif env.unwrapped.envdict['action_space'] == 'STATEMACHINE':
        action = smaction(keyaction)
        keyaction='0'

    state, reward, terminated, cause=env.step(action)

    if i%200==0:
        plt.close('all')
    action = 12#np.random.randint(0,25)
    if terminated:
        print(state)
        print(reward)
        print(cause)
        #action = 12
        env._reset()# some comment wee added
elapsed=time.time()-t
print(elapsed)
print('DONE')