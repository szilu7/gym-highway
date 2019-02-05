import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from gym_highway.modell.modell import Modell
from gym_highway.modell.modell import Egovehicle
from gym_highway.modell.modell import Envvehicle
from gym_highway.modell.environment_vehicle import CollisionExc
import os
import datetime

logger = logging.getLogger(__name__)
"""
   The following section provides global parameters, to modify the environment
   For different types of agent behavior, modify 
   globals['action_space'] ('DISCRETE' 'CONTINUOUS' 'STATEMACHINE')
"""
globals = {}
#Highway geometry parameters
globals['length_forward']= 1000  # Simulation scope in  front of the ego vehicle in [m]
globals['length_backward']= 500  # Simulation scope behind the ego vehicle in [m]
globals['dt'] = 0.2              # Steptime in [s]
globals['lane_width'] = 4        # Width of one Lane in [m]
globals['lane_count'] = 3        # Number of lanes

# Agent action parameters
globals['action_space']= 'STATEMACHINE'    # agent action type. Choose from: 'DISCRETE' 'CONTINUOUS' 'STATEMACHINE'

# parameters for DISCRETE action. Sets of steering angles (st in[rad]) and accelerations ([ac in m/s^2]) to chose from
globals['discreteparams']={'st' : [-0.003, -0.0005, 0, 0.0005, 0.003],'ac' : [-6.0, -2.0, 0.0, 2.0, 3.5] }
# parameters for CONTINUOUS action. Lower and upper bounds (alow,ahigh)[steering [rad],acceleration [m/s^2]]
globals['continuousparams']= {'alow' : np.array([-0.003, -6.0]), 'ahigh' : np.array([0.003, 3.5])}

# Vehicle Generation Parameters
globals['density_lanes_LB'] = 18   # Lower bound of the random density for one lane [vehicle/km]
globals['density_lanes_UB'] = 26  # Upper bound of the random density for one lane [vehicle/km]

# Vehicle desired speed generation parameters normal distribution N(mean,sigma^2) in [m/s] for each lane
# IMPORTANT! Need to add as many values, as globals['lane_count']
globals['speed_lane0'] = [30.0, 9.0]  # generated vehicle desired speed lane 0 [m/s]
globals['speed_lane1'] = [35.0, 9.0]  # generated vehicle desired speed lane 1 [m/s]
globals['speed_lane2'] = [40.0, 9.0]  # generated vehicle desired speed lane 2 [m/s]

# Agent vehicle desired Speed
globals['speed_ego_desired'] = 130.0/3.6  # Agent vehicle desired speed [m/s]


# Subreward Weights
globals['creward'] = 0.3 # Subreward weight for distances to other vehicles
globals['lreward'] = 0.3 # Subreward weight for keeping right behavior
globals['yreward'] = 0.1 # Subreward weight for (not) leaving highway
globals['vreward'] = 0.3 # Subreward weight for keeping desired speed


"""
globals['creward'] = 0.05 # Subreward weight for distances to other vehicles
globals['lreward'] = 0.85 # Subreward weight for keeping right behavior
globals['yreward'] = 0.05 # Subreward weight for (not) leaving highway
globals['vreward'] = 0.05 # Subreward weight for keeping desired speed
"""

class EPHighWayEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):

        self.envdict=globals
        self.modell = None

        #Set action space
        if self.envdict['action_space']=='CONTINUOUS':
            params = self.envdict['continuousparams']
            self.action_space = spaces.Box( params['alow'], params['ahigh'])
            self.actiontype=0
        elif self.envdict['action_space']=='DISCRETE':
            params=self.envdict['discreteparams']
            self.st=params['st']
            self.stlen=len(self.st)
            self.ac=params['ac']
            self.aclen=len(self.ac)
            self.action_space = spaces.Discrete(self.stlen*self.aclen)
            self.actiontype = 1
        elif self.envdict['action_space']=='STATEMACHINE':
            self.action_space = spaces.Discrete(3)
            self.actiontype=2
        else:
            raise (Exception('Unknown action_space'))
        self.envdict['actiontype']=self.actiontype

        #Set observation space
        low = np.array([0, -50, 0, -50, 0, -50, 0, -50, 0, -50, 0, -50, 0, 0, -5, -.5, 0])
        high = np.array([500, 50, 500, 50, 500, 50, 500, 50, 500, 50, 500, 50, 1, 1, 25, .5, 50])
        self.observation_space = spaces.Box(low, high)

        self.rewards = [0, 0, 0, 0]
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.envdict['density_lanes'] = np.random.randint(self.envdict['density_lanes_LB'] , self.envdict['density_lanes_UB'] )  # [vehicle density vehicle/km]

        self.modell = Modell(self.envdict)
        self.modell.warmup(False)
        # Picking the EgoVehicle from modell
        self.modell.searchEgoVehicle()

        self.rewards = [0., 0., 0., 0.]
        # Aquiring state from modell
        self.state = self.modell.generate_state_for_ego()
        return self.state

    def calcaction(self, action):
        if self.actiontype==0:
            return action
        if self.actiontype==1:
            steer = self.st[action // self.aclen]
            acc = self.ac[action % self.aclen]
            ctrl = [steer, acc]
            return ctrl
        if self.actiontype==2:
            return action

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        ctrl = self.calcaction(action)
        if self.actiontype!=2:
            isOk, cause = self.modell.onestep(ctrl)
            self.state = self.modell.generate_state_for_ego()
        else:
            while True:
                isOk, cause = self.modell.onestep(ctrl)
                self.state = self.modell.generate_state_for_ego()
                if (not isOk) or (self.modell.egovehicle.state=='command_receive'):
                    break
        terminated = not isOk
        if terminated:
            print(cause)
            reward = -80.0
            rewards = np.zeros(4)
        else:
            reward, rewards = self.calcreward()

        return self.state, reward, terminated, {'cause': cause, 'rewards': self.rewards}

    def calcreward(self):
        reward = 0
        # LANE BASED REWARD
        lreward = 0
        laneindex = self.modell.egovehicle.laneindex

        if laneindex > 0:
            if (self.state[13] == 0) and (self.state[4] > 30):
                lreward = -min(1, max(0, (self.state[4] - 50.0) / 20.0))
        lreward += 1.


        """
        if laneindex > 0:
            if (self.state[13] == 0) and (self.state[4] > 30):
                lreward = (-min(1, max(0, (self.state[4] - 50.0) / 20.0))) * 3
        lreward += 1.
        """

        # POSITION BASED REWARD
        # dy=abs(self.modell.egovehicle.y-laneindex*self.envdict['lane_width'])
        dy = abs(self.modell.egovehicle.y - (self.envdict['lane_count'] - 1.0) / 2 * self.envdict['lane_width'])
        ytresholdlow = (self.envdict['lane_count'] - 1.0) / 2.0 * self.envdict['lane_width'] + 0.3  # [m]
        ytresholdhigh = (self.envdict['lane_count']) / 2.0 * self.envdict['lane_width']  # [m]
        yrewhigh = 1.0
        yrewlow = 0.0
        if dy < ytresholdlow:
            yreward = yrewhigh
        elif dy > ytresholdhigh:
            yreward = yrewlow
        else:
            yreward = yrewhigh - (yrewhigh - yrewlow) * (dy - ytresholdlow) / (ytresholdhigh - ytresholdlow)
        # DESIRED SPEED BASED REWARD
        dv = abs(self.modell.egovehicle.desired_speed - self.state[16])
        vtresholdlow = 2  # [m/s]
        vtresholdhigh = 25  # self.modell.egovehicle.desired_speed #[m/s]
        vrewhigh = 1.0
        vrewlow = 0.1
        if dv < vtresholdlow:
            vreward = vrewhigh
        elif dv > vtresholdhigh:
            vreward = vrewlow
        else:
            vreward = vrewhigh - (vrewhigh - vrewlow) * (dv - vtresholdlow) / (vtresholdhigh - vtresholdlow)

        # Vehicle Closing Based Rewards
        cright = 0  # right safe zone
        cleft = 0  # left safe zone
        cfront = 0  # followed vehicle
        crear = 0  # following vehicle

        lw = self.envdict['lane_width']
        vehy = self.modell.egovehicle.y - self.modell.egovehicle.laneindex * lw

        # right safe zone
        if self.state[13] == 1:
            if vehy < -lw / 4:
                cright = max(-1, (vehy + lw / 4) / (lw / 4))
        # left safe zone
        if self.state[12] == 1:
            if vehy > lw / 4:
                cleft = max(-1, -(vehy - lw / 4) / (lw / 4))
        # front
        followingtime = self.state[2] / self.state[16]
        if followingtime < 1:
            cfront = followingtime - 1
        # rear
        followingtime = self.state[8] / self.state[16]
        if followingtime < 0.5:
            crear = (followingtime - 0.5) * 2

        creward = 1 + max(-1, min(cright , cleft , cfront , crear))
        creward *= self.envdict['creward']
        lreward *= self.envdict['lreward']
        yreward *= self.envdict['yreward']
        vreward *= self.envdict['vreward']

        reward = lreward + yreward + vreward + creward
        rewards = {'y': yreward, 'v': vreward, 'l': lreward, 'c': creward}

        self.rewards[0] += rewards['l']
        self.rewards[1] += rewards['y']
        self.rewards[2] += rewards['v']
        self.rewards[3] += rewards['c']

        return reward, rewards

    def _render(self, mode='human', close=False):
        self.modell.render(True, self.rewards)
