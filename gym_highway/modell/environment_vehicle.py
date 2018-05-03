from gym_highway.modell.vehicle_base import BaseVehicle
import numpy as np


class Envvehicle(BaseVehicle):

    def __init__(self, dict):
        super().__init__(dict)
        self.desired_speed = 0.0
        self.maxacc = 2.0  # Max acceleration m/s^2
        self.maxdec = -6.0  # Max deceleration m/s^2
        self.state = 'in_lane'
        self.change_needed = 0
        self.change_finished = 0  # Indicates the the vehicle is leaving its lane
        self.laneindex = 0
        self.lanewidth = self.envdict['lane_width']
        self.oldlane = 0
        self.skip = 0

    def _getrandom(self):
        sigma = 5
        self.desired_speed = 130 / 3.6 + sigma * np.random.randn()
        self.vx = self.desired_speed

    def step(self, vnext, vbehind, vright_a, vright_b, vleft_a, vleft_b):
        """
        Steps with vehicle. Updates state

        decision:  1- Reach target speed
                   2- Follow next vehicle

        :param vnext: vehicle in front
        :param vright: vehicle to the right
        :param vleft: vehicle to the left
        :return: Vehicle reached highway limit (front or rear)
        """
        if self.state == 'in_lane':
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
                    raise CollisionExc('Collision')
                # desired following dist
                dist = vnext.vx * 1.4
                ddist = dist - dx
                accghr = -1 * ddist + 10 * dv

                # alpha=0.6
                # m=0.4
                # l=1.9
                # accghr=alpha*(self.vx**m)*dv/(dx**l)

                accghr = min(max(self.maxdec, accghr), self.maxacc)
                if self.vx > self.desired_speed:
                    acc = min(-self.maxacc, accghr)
                else:
                    acc = accghr

            self.vx = self.vx + self.dt * acc
            self.x = self.x + self.dt * self.vx

            #  Jobbratartás
            if (self.laneindex != 0):
                if (vright_a is None) or (((vright_a.x - vright_a.length - self.x) / 11) > self.length):
                    if (vright_b is None) or (((self.x - self.length - vright_b.x) / 9) > self.length):
                        if not (vright_a is None):
                            if ((self.vx * 0.7) < vright_a.vx):
                                self.state = 'switch_lane_right'
                                self.change_needed = 1
                                self.oldlane = self.laneindex
                        else:
                            self.state = 'switch_lane_right'
                            self.change_needed = 1
                            self.oldlane = self.laneindex

            #  Feltartja a mögötte haladót, lehúzódás jobbra
            """
            if not (vbehind is None):
                if (
                        self.x - self.length - vbehind.x) / 5 < self.length:  # Mögötte haladó megközelítette 5 autónyi távolságra
                    if (vbehind.desired_speed > self.desired_speed):
                        if (vright_a is not None) and (vright_b is not None):
                            if ((
                                        vright_a.x - vright_a.length - self.x) / 5) > self.length:  # Előtte 5 autónyi hely van a jobb oldali sávban
                                if ((
                                            self.x - vright_b.x) / 8) > self.length:  # Mögötte 8 autónyi hely van a jobb oldali sávban
                                    print("Under overtake")
                                    #self.state = 'switch_lane_right'
                                    #self.switch_lane = 1
            """
            #  Gyorsabban menne, előzés

            if self.laneindex != (self.envdict['lane_count'] - 1):
                if not (vnext is None):
                    diff = (vnext.x - vnext.length - self.x)
                    if ((diff / 9) < self.length):
                        if self.desired_speed > vnext.desired_speed:
                            if (vleft_a is None) or (((vleft_a.x - vleft_a.length - self.x) / 4) > self.length):
                                if (vleft_b is None) or (((self.x - self.length - vleft_b.x) / 4) > self.length):
                                        if (vbehind is None) or (isinstance(vbehind, Envvehicle) and (vbehind.state != 'acceleration')):
                                            self.state = 'acceleration'
                                            s = vnext.x - vnext.length - self.x
                                            vrel = abs(vnext.vx - self.vx)
                                            t = 3 / self.envdict['dt']
                                            a = abs((2 * (s - (vrel * t))) / (t * t))
                                            self.maxacc = a

        elif self.state == 'switch_lane_right':
            acc = self.maxacc

            if not (vnext is None):
                # following GHR model
                dv = vnext.vx - self.vx
                dx = vnext.x - vnext.length - self.x
                if dx < 0:
                    print('Collision, ID: ', self.ID, ' vnext ID: ', vnext.ID, ' in lane: ', self.laneindex)
                    raise CollisionExc('Collision')
                # desired following dist
                dist = vnext.vx * 1.4
                ddist = dist - dx
                accghr = -1 * ddist + 10 * dv

                accghr = min(max(self.maxdec, accghr), self.maxacc)
                if self.vx > self.desired_speed:
                    acc = min(-self.maxacc, accghr)
                else:
                    acc = accghr

            self.vx = self.vx + self.dt * acc
            self.x = self.x + self.dt * self.vx

            self.y = self.y - 0.4
            if self.y <= ((self.laneindex - 1) * self.lanewidth):
                self.y = ((self.laneindex - 1) * self.lanewidth)
                self.laneindex = self.laneindex - 1
                self.change_finished = 1
                self.state = 'in_lane'

        elif self.state == 'switch_lane_left':
            acc = max(self.maxacc, 2)
            if not (vnext is None):
                # following GHR model
                dv = vnext.vx - self.vx
                dx = vnext.x - vnext.length - self.x
                if dx < 0:
                    print('Collision, ID: ', self.ID, ' vnext ID: ', vnext.ID, ' in lane: ', self.laneindex)
                    raise CollisionExc('Collision')

                # desired following dist
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

            self.y = self.y + 0.4
            if self.y >= ((self.laneindex + 1) * self.lanewidth):
                self.y = ((self.laneindex + 1) * self.lanewidth)
                self.laneindex = self.laneindex + 1
                self.state = 'in_lane'

        elif self.state == 'acceleration':
            acc = self.maxacc

            self.vx = self.vx + self.dt * acc
            self.x = self.x + self.dt * self.vx

            if not (vnext is None):
                s = (vnext.x - vnext.length - self.x)
                if (s / 3) < self.length:
                    if not (vleft_b is None):
                        if (self.vx > (0.8 * vleft_b.vx)) and (((self.x - self.length - vleft_b.x) / 3) > self.length):
                            if not (vleft_a is None):
                                if (((vleft_a.x - vleft_a.length - self.x) / 3) > self.length) and \
                                        (vleft_a.vx > (self.x *0.8)):
                                    self.state = 'switch_lane_left'
                                    self.change_needed = 1
                                    self.maxacc = 2
                                    self.oldlane = self.laneindex
                                    print('Overtake at: ', self.x)
                                else:
                                    self.state = 'in_lane'
                                    self.vx = vnext.vx
                            else:
                                self.state = 'switch_lane_left'
                                self.change_needed = 1
                                self.maxacc = 2
                                self.oldlane = self.laneindex
                                print('Overtake at: ', self.x)
                        else:
                            self.state = 'in_lane'
                            self.vx = vnext.vx
                    else:
                        if not (vleft_a is None):
                            if (((vleft_a.x - vleft_a.length - self.x) / 3) > self.length):
                                self.state = 'switch_lane_left'
                                self.change_needed = 1
                                self.maxacc = 2
                                self.oldlane = self.laneindex
                                print('Overtake at: ', self.x)
                            else:
                                self.state = 'in_lane'
                                self.vx = vnext.vx
                        else:
                            self.state = 'switch_lane_left'
                            self.change_needed = 1
                            self.maxacc = 2
                            self.oldlane = self.laneindex
                            print('Overtake at: ', self.x)
            else:
                self.state = 'in_lane'
        reachedend = False

        if (self.x > self.envdict['length_forward']) or (self.x < self.envdict['length_backward']):
            reachedend = True

        return reachedend

    def warmup_step(self, vnext):
        """
        Steps with vehicle. Updates state

        decision:  1- Reach target speed
                   2- Follow next vehicle

        :param vnext: vehicle in front
        :param vright: vehicle to the right
        :param vleft: vehicle to the left
        :return: Vehicle reached highway limit (front or rear)
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
                raise CollisionExc('Collision')
            # desired following dist
            dist = vnext.vx * 1.4
            ddist = dist - dx
            accghr = -1 * ddist + 10 * dv

            # alpha=0.6
            # m=0.4
            # l=1.9
            # accghr=alpha*(self.vx**m)*dv/(dx**l)

            accghr = min(max(self.maxdec, accghr), self.maxacc)
            if self.vx > self.desired_speed:
                acc = min(-self.maxacc, accghr)
            else:
                acc = accghr

        self.vx = self.vx + self.dt * acc
        self.x = self.x + self.dt * self.vx


class CollisionExc(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)