"""
    Base Vehicle Class
    Vehicle State is vector
        0 - Position longitudnal [m]
        1 - Position lateral [m]
        2 - Heading (dir->x =0, CCW) [rad]
        3 - Speed x direction [m/s]
        4 - Speed y direction [m/s]
"""
import numpy as np
import matplotlib.pyplot as plt

class BaseVehicle:
    def __init__(self,dict):
        self.envdict=dict
        self.dt=self.envdict['dt']
        self.length=5 #vehicle length in [m]
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.color='b'
        self.ID = 0

    def render(self):
        x=self.x
        y=self.y
        l=self.length
        plt.plot([x,x,x+l,x+l,x],[y-1,y+1,y+1,y-1,y-1],self.color)





