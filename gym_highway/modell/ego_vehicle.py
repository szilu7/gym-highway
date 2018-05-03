from gym_highway.modell.vehicle_base import BaseVehicle
import numpy as np
import math


class Egovehicle(BaseVehicle):
    def __init__(self, dict):
        super().__init__(dict)
        self.desired_speed = dict['speed_ego_desired']
        self.actiontype = dict['actiontype']

        self.maxacc = 2.0  # Max acceleration m/s^2
        self.maxdec = -6.0  # Max deceleration m/s^2
        self.color = 'r'
        self.laneindex = 0
        # Properties for state based behavior
        if self.actiontype==2:
            self.state = 'command_receive'
            self.cmd = 'in_lane'
            self.change_needed = 0
            self.change_finished = 0
            self.lanewidth = self.envdict['lane_width']
            self.oldlane = 0
        else:
            self.change_needed = 0
            self.state=''



    def vehicle_onestep(self, vehiclestate, action, dt):
        """
        :param vehiclestate: np.array([x,y,th,v])
                            x,y - position ([m,m])
                            th  - angle ([rad] zero at x direction,CCW)
                            v   - velocity ([m/s])
        :param action: np.array([steering, acceleration])
                            steering     - angle CCW [rad]
                            acceleration - m/s^2
        :param dt: sample time [s]
        :return:the new vehicle state in same structure as the vehiclestate param
        """
        # Fixed Vehicle axle length
        L = 3

        state = vehiclestate
        # The new speed v'=v+dt*a
        state[3] = max(0, vehiclestate[3] + dt * action[1])
        # The travelled distance s=(v+v')/2*dt
        s = (state[3] + vehiclestate[3]) / 2 * dt

        if action[0] == 0:  # Not steering
            # unit vector
            dx = math.cos(state[2])
            dy = math.sin(state[2])
            state[0] = vehiclestate[0] + dx * s
            state[1] = vehiclestate[1] + dy * s
        else:  # Steering
            # Turning Radius R=axlelength/tanh(steering)
            R = L / math.tanh(action[0])
            # The new theta heading th'=th+s/R
            turn = s / R
            state[2] = vehiclestate[2] + turn
            if math.pi < state[2]:
                state[2] = state[2] - 2 * math.pi
            if -math.pi > state[2]:
                state[2] = state[2] + 2 * math.pi
            # new position
            # transpose distance dist=2*R*sin(|turn/2|)
            dist = abs(2 * R * math.sin(turn / 2))
            # transpose angle ang=th+turn/2
            ang = vehiclestate[2] + turn / 2
            # unit vector
            dx = math.cos(ang)
            dy = math.sin(ang)
            # new position
            state[0] = vehiclestate[0] + dx * dist
            state[1] = vehiclestate[1] + dy * dist
        return state

    def step(self, action, vnext):
        if self.actiontype==2:
            self.step_SM(action,vnext)
        else:
            self.step_CONT(action)


    def step_CONT(self, action):
        th = math.atan2(self.vy, self.vx)
        v = math.sqrt(self.vx ** 2 + self.vy ** 2)
        # print(th,v)
        state = np.array([self.x, self.y, th, v])
        newstate = self.vehicle_onestep(state, action, self.envdict['dt'])
        self.x = newstate[0]
        self.y = newstate[1]
        self.vx = newstate[3] * math.cos(newstate[2])
        self.vy = newstate[3] * math.sin(newstate[2])


    def step_SM(self, action, vnext):
        """
        th=math.atan2(self.vy,self.vx)
        v=math.sqrt(self.vx**2+self.vy**2)
        #print(th,v)
        state=np.array([self.x, self.y, th, v])
        newstate=self.vehicle_onestep(state,action,self.envdict['dt'])
        self.x=newstate[0]
        self.y = newstate[1]
        self.vx = newstate[3]*math.cos(newstate[2])
        self.vy = newstate[3] * math.sin(newstate[2])
        """

        if (self.state == 'command_receive'):
            if action == 0:    #
                self.cmd = 'switch_lane_left'
                self.change_needed = 1
                self.oldlane = self.laneindex
                self.state = 'command_execute'
                self.switch_lane_left()

            elif action == 1:  #
                self.cmd = 'switch_lane_right'
                self.change_needed = 1
                self.oldlane = self.laneindex
                self.state = 'command_execute'
                self.switch_lane_right()

            else:
                self.in_lane(vnext)

        elif (self.state == 'command_execute'):
            if (self.cmd == 'switch_lane_right'):
                self.switch_lane_right()
            elif (self.cmd == 'switch_lane_left'):
                self.switch_lane_left()

    def in_lane(self, vnext):
        """
        # Desired acceleration
        if self.vx < self.desired_speed:
            acc = self.maxacc
        else:
            acc = -self.maxacc

        self.vx = self.vx + self.dt * acc
        self.x = self.x + self.dt * self.vx
        """

        acc = 0
        # Desired acceleration
        if self.vx < self.desired_speed:
            acc = self.maxacc
        else:
            acc = -self.maxacc

        if not (vnext is None):
            # following GHR model
            dv = vnext.vx - self.vx
            dx = vnext.x - vnext.length - self.x
            if dx < 0:
                print('Collision, ID: ', self.ID, ' vnext ID: ', vnext.ID, ' in lane: ', self.laneindex)
                print(vnext.x, ' - ', vnext.length, ' - ', self.x)

            dist = vnext.vx * 1.0
            ddist = dist - dx
            accghr = -1 * ddist + 10 * dv
            accghr = min(max(self.maxdec, accghr), self.maxacc)
            if self.vx > self.desired_speed:
                acc = min(-self.maxacc, accghr)
            else:
                acc = accghr

        self.vx = self.vx + self.dt * acc
        self.x = self.x + self.dt * self.vx

    def switch_lane_right(self):
        self.x = self.x + self.dt * self.vx
        self.y = self.y - 0.4

        if self.y <= ((self.oldlane - 1) * self.lanewidth):
            self.y = ((self.oldlane - 1) * self.lanewidth)
            self.state = 'command_receive'


    def switch_lane_left(self):

        self.x = self.x + self.dt * self.vx
        self.y = self.y + 0.4

        if self.y >= ((self.oldlane + 1) * self.lanewidth):
            self.y = ((self.oldlane + 1) * self.lanewidth)
            self.state = 'command_receive'
