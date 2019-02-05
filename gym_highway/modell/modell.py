import gym
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from gym_highway.modell.ego_vehicle import Egovehicle
from gym_highway.modell.environment_vehicle import Envvehicle, env_add_entry


class Modell:
    def __init__(self, envdict):
        self.envdict = envdict
        self.egovehicle = None

        self.highwaylength = self.envdict['length_forward'] + self.envdict['length_backward']
        self.lanes = []
        self.nextvehicle = []
        for i in range(self.envdict['lane_count']):
            self.lanes.append([])

        self.prev_data = []
        self.actual_data = []
        self.id = 0

        """
        self.log_list = []
        self.log_cnt = 0
        self.logs_in_file = 15
        """

        self.log = Envvehicle(self.envdict)

    def onestep(self, action):
        """
        :param action: takes action for egovehicle
        :return: success, cause
        """


        # 5. NewBorn vehicles If lane density is smaller than desired
        self.generate_new_vehicles()
        self.random_new_des_speed()


        # 1. Stepping the ego vehicle
        vehiclecnt = len(self.lanes[self.egovehicle.laneindex])
        egoveh_lane = self.lanes[self.egovehicle.laneindex]
        vnext = None

        for i in range(vehiclecnt):
            if isinstance(egoveh_lane[i], Egovehicle):
                pos = i
                break
        if (i + 1) <vehiclecnt:
            vnext = egoveh_lane[i + 1]

        self.egovehicle.step(action, vnext)
        if self.egovehicle.vx < 10:
            return False, 'Low speed'
        # # perform lane change and collision check
        # fine, cause = self.check_position()
        # if not fine:
        #     return False, cause

        # 2. Transpose everyone to set x of egovehicle to 0
        offs = -self.egovehicle.x
        for j in range(self.envdict['lane_count']):
            lane = self.lanes[j]
            for i in range(len(lane)):
                lane[i].x = lane[i].x + offs
        # 3. Stepping every other vehicle,
        for i in range(self.envdict['lane_count']):
            lane = self.lanes[i]
            for j in range(len(lane)):
                vehiclecnt = len(lane)
                if j < vehiclecnt:
                    veh = lane[j]
                    if isinstance(veh, Envvehicle):
                        if (veh.skip == 0):
                            if j + 1 < vehiclecnt:
                                vnext = lane[j + 1]
                            else:
                                vnext = None

                            if j - 1 >= 0:
                                vbehind = lane[j - 1]
                            else:
                                vbehind = None

                            vright_a = None
                            vright_b = None
                            if i > 0:
                                lane_right = self.lanes[i - 1]
                                for k in range(len(lane_right)):
                                    if lane_right[k].x > veh.x:
                                        vright_a = lane_right[k]
                                        break
                                for k in range(len(lane_right)-1, -1, -1):    # from len(lane_right) to -1 with -1, 0 is the last!
                                    if lane_right[k].x < veh.x:
                                        vright_b = lane_right[k]
                                        break

                            vleft_a = None
                            vleft_b = None
                            if i < self.envdict['lane_count'] - 1:
                                lane_left = self.lanes[i + 1]
                                for k in range(len(lane_left)):
                                    if lane_left[k].x > veh.x:
                                        vleft_a = lane_left[k]
                                        break
                                for k in range(len(lane_left)-1, -1, -1):
                                    if lane_left[k].x < veh.x:
                                        vleft_b = lane_left[k]
                                        break

                            veh.step(vnext, vbehind, vright_a, vright_b, vleft_a, vleft_b)

                            if veh.state == 'switch_lane_right':
                                oldlane = self.lanes[veh.oldlane]
                                for l in range(len(oldlane)):
                                    if oldlane[l].ID == veh.ID:
                                        oldlane[l].x = veh.x
                                        oldlane[l].vx = veh.vx

                            if (veh.change_finished == 1):
                                veh.change_finished = 0
                                oldlane = self.lanes[veh.oldlane]
                                veh.state = 'in_lane'
                                for vehicle in oldlane:
                                    if vehicle.ID == veh.ID:
                                        oldlane.remove(vehicle)
                                        #print('Removed in Step ID:  ' + str(veh.ID))
                                        break

        #3.5 Insert vehicle in the other lane if lane switch has occurred
        for i in range(self.envdict['lane_count']):
            lane = self.lanes[i]
            for j in range(len(lane)):
                if j < len(lane):
                    veh = lane[j]
                    if isinstance(veh, Envvehicle):
                        if (veh.state == 'switch_lane_right') and (veh.change_needed == 1):
                            newlane = self.lanes[i - 1]
                            veh.skip = 1
                            ev = Envvehicle(self.envdict)
                            inserted = 0

                            vright_a = None
                            vright_b = None
                            if i > 0:
                                lane_right = self.lanes[i - 1]
                                for k in range(len(lane_right)):    # search the vehicle ahead in the right lane
                                    if lane_right[k].x > veh.x:
                                        vright_a = lane_right[k]
                                        break
                                for k in range(len(lane_right) - 1, -1, -1):  # from len(lane_right) to -1 with -1, 0 is the last!
                                    if lane_right[k].x < veh.x:     # search the vehicle behind in the right lane
                                        vright_b = lane_right[k]
                                        break

                            if (vright_a is None) or ((vright_a.state != 'switch_lane_left') and
                                                      (vright_a.state != 'acceleration')):
                                if (vright_b is None) or ((vright_b.state != 'switch_lane_left') and
                                                       (vright_b.state != 'acceleration')):
                                    for k in range(len(newlane)):
                                        if (newlane[k].x > veh.x):
                                            veh.change_needed = 0
                                            ev=copy.copy(veh)

                                            ev.skip = 0
                                            newlane.insert(k, ev)
                                            inserted = 1
                                            break
                                    if inserted == 0:
                                        veh.change_needed = 0
                                        ev = copy.copy(veh)

                                        ev.skip = 0
                                        newlane.insert(len(newlane), ev)
                                else:
                                    veh.state = 'in_lane'
                                    veh.change_needed = 0
                                    veh.skip = 0
                            else:
                                veh.state = 'in_lane'
                                veh.change_needed = 0
                                veh.skip = 0

                        elif (veh.state == 'switch_lane_left') and (veh.change_needed == 1):
                            newlane = self.lanes[i + 1]

                            vleft_a = None
                            if i < self.envdict['lane_count']:
                                lane_left = self.lanes[i + 1]
                                for k in range(len(lane_left)):
                                    if lane_left[k].x > veh.x:
                                        vleft_a = lane_left[k]
                                        break

                            if (vleft_a is None) or ((vleft_a.state != 'switch_lane_right')):
                                for k in range(len(newlane)):
                                    if (newlane[k].x > veh.x):
                                        veh.change_needed = 0
                                        newlane.insert(k, veh)
                                        break
                                oldlane = self.lanes[veh.oldlane]
                                for vehicle in oldlane:
                                    if vehicle.ID == veh.ID:
                                        oldlane.remove(vehicle)
                                        break
                            elif not (vleft_a is None) and ((vleft_a.state == 'switch_lane_right')):
                                if ((vleft_a.x - vleft_a.length - veh.x) / 4) > veh.length:
                                    for k in range(len(newlane)):
                                        if (newlane[k].x > veh.x):
                                            veh.change_needed = 0
                                            newlane.insert(k, veh)
                                            break
                                    oldlane = self.lanes[veh.oldlane]
                                    for vehicle in oldlane:
                                        if vehicle.ID == veh.ID:
                                            oldlane.remove(vehicle)
                                            break
                            else:
                                veh.state = 'in_lane'
                                print('Acceleration left in MODEL ID: ' + str(veh.ID))
                                veh.change_needed = 0

                        elif (veh.change_finished == 1):
                            veh.change_finished = 0
                            oldlane = self.lanes[veh.oldlane]
                            veh.state = 'in_lane'
                            removed = 0
                            for vehicle in oldlane:
                                if vehicle.ID == veh.ID:
                                    oldlane.remove(vehicle)
                                    removed = 1
                                    break
                            if removed == 0:
                                for m in range(self.envdict['lane_count']):
                                    lane = self.lanes[m]
                                    for n in range(len(lane)):
                                        if (lane[n].ID == veh.ID) and (lane[n].state != 'in_lane'):
                                            lane.remove(lane[n])
                                            removed = 1
                                            print('Removed for 2nd try ID:  ' + str(veh.ID))
                            if removed == 0:
                                print('NOT removed ID:  ' + str(veh.ID))



                    elif isinstance(veh, Egovehicle) and (veh.change_needed == 1):
                        inserted = 0
                        if veh.cmd == 'switch_lane_left':
                            if (i + 1) < self.envdict['lane_count']:
                                newlane = self.lanes[i + 1]
                                for k in range(len(newlane)):
                                    if (newlane[k].x > veh.x):
                                        newlane.insert(k, veh)
                                        inserted = 1
                                        break
                                if inserted == 0:
                                    newlane.insert(len(newlane), veh)
                                oldlane = self.lanes[veh.oldlane]
                                for vehicle in oldlane:
                                    if vehicle.ID == veh.ID:
                                        oldlane.remove(vehicle)
                                        break
                                veh.laneindex = veh.laneindex + 1

                        elif veh.cmd == 'switch_lane_right':
                            newlane = self.lanes[i - 1]
                            for k in range(len(newlane)):
                                if (newlane[k].x > veh.x):
                                    newlane.insert(k, veh)
                                    inserted = 1
                                    break
                            if inserted == 0:
                                newlane.insert(len(newlane), veh)
                            oldlane = self.lanes[veh.oldlane]
                            for vehicle in oldlane:
                                if vehicle.ID == veh.ID:
                                    oldlane.remove(vehicle)
                                    break
                            veh.laneindex = veh.laneindex - 1
                        veh.change_needed = 0

        # 4. Deleting vehicles out of range
        for j in range(self.envdict['lane_count']):
            lane = self.lanes[j]
            for veh in lane:
                if veh.x < -self.envdict['length_backward'] or veh.x > self.envdict['length_forward']:
                    if veh.state == 'switch_lane_right':
                        oldlane = self.lanes[veh.oldlane]
                        for vehicle in oldlane:
                            if vehicle.ID == veh.ID:
                                oldlane.remove(vehicle)
                    #print("ID: " + str(veh.ID) + " laneindex: " + str(veh.laneindex) + " oldlane: " + str(veh.oldlane))
                    try:
                        lane.remove(veh) #ValueError: list.remove(x): x not in list
                    except ValueError:
                        print("x not in list")

        # 4.5 Recheck position
        fine, cause = self.check_position()
        if not fine:
            return False, cause

        """"
        # 5. NewBorn vehicles If lane density is smaller than desired
        self.generate_new_vehicles()
        self.random_new_des_speed()
        """


        """
        # 6. Save data
        file = open(r'D:\file.txt', 'a')
        file.write('New cycle\n')
        for i in range(self.envdict['lane_count']):
            lane = self.lanes[i]
            for j in range(len(lane)):
                lane_num = i
                index = lane[j].laneindex
                x = lane[j].x
                y = lane[j].y
                vx = lane[j].vx
                id = lane[j].ID
                if lane[j].x != 0:
                    st = lane[j].state
                    text = 'lane: '+str(lane_num)+' idx: '+str(index)+' ID: ' +str(id)+' x: '+ str(x)+\
                           ' y: '+str(y)+' vx: '+str(vx)+' '+st+' skipped: '+ str(lane[j].skip) + '\n'
                else:
                    text = 'lane: '+str(lane_num)+' idx: '+str(index)+' ID: '+str(id)+' x: '+str(x)+\
                           ' y: '+str(y)+' vx: '+str(vx)+'\n'
                file.write(text)
        file.close()
        """

        # 6. Save data
        """
        self.log_cnt += 1
        write = 'Step ' + str(self.log_cnt) + '\n'
        for i in range(self.envdict['lane_count']):
            lane = self.lanes[i]
            for j in range(len(lane)):
                lane_num = i
                index = lane[j].laneindex
                x = lane[j].x
                y = lane[j].y
                vx = lane[j].vx
                id = lane[j].ID
                if lane[j].x != 0:
                    st = lane[j].state
                    text = 'lane: '+str(lane_num)+' idx: '+str(index)+' ID: ' +str(id)+' x: '+ str(x)+\
                           ' y: '+str(y)+' vx: '+str(vx)+' '+st+' skipped: '+ str(lane[j].skip) + '\n'
                else:
                    text = 'lane: '+str(lane_num)+' idx: '+str(index)+' ID: '+str(id)+' x: '+str(x)+\
                           ' y: '+str(y)+' vx: '+str(vx)+'\n'
                write += text
        self.log_list.append(write)

        if self.log_cnt > self.logs_in_file:
            self.log_list.pop(0)

        self.save_log()
        """

        write = ""
        for i in range(self.envdict['lane_count']):
            lane = self.lanes[i]
            for j in range(len(lane)):
                lane_num = i
                index = lane[j].laneindex
                x = lane[j].x
                y = lane[j].y
                vx = lane[j].vx
                id = lane[j].ID
                if lane[j].x != 0:
                    st = lane[j].state
                    text = 'lane: '+str(lane_num)+' idx: '+str(index)+' ID: ' +str(id)+' x: '+ str(x)+\
                           ' y: '+str(y)+' vx: '+str(vx)+' '+st+' skipped: '+ str(lane[j].skip) + '\n'
                else:
                    text = 'lane: '+str(lane_num)+' idx: '+str(index)+' ID: '+str(id)+' x: '+str(x)+\
                           ' y: '+str(y)+' vx: '+str(vx)+'\n'
                write += text
        env_add_entry(write)





        return True, 'Fine'


    def check_position(self):
        laneindex = int(round(self.egovehicle.y / self.envdict['lane_width']))
        if (laneindex < 0) or (laneindex + 1 > self.envdict['lane_count']):
            return False, 'Left Highway'

        """
        if (self.envdict['actiontype']!=2) and (laneindex != self.egovehicle.laneindex):
            # sávváltás van
            oldlane = self.lanes[self.egovehicle.laneindex]
            newlane = self.lanes[laneindex]
            oldlane.remove(self.egovehicle)
            i = 0
            for i in range(len(newlane)):
                if newlane[i].x > self.egovehicle.x:
                    break
            newlane.insert(i, self.egovehicle)
            self.egovehicle.laneindex = laneindex
        """
        lane = self.lanes[self.egovehicle.laneindex]
        i = lane.index(self.egovehicle)
        front = i + 1
        if len(lane) > front:
            if self.egovehicle.x > lane[front].x - lane[front].length:
                return False, 'Front Collision'

        rear = i - 1
        if 1 <= i:
            if lane[rear].x > self.egovehicle.x - self.egovehicle.length:
                #print('ID: '+str(lane[rear].ID)+' x: '+str(lane[rear].x))
                return False, 'Rear Collision'

        return True, 'Everything is fine'

    def generate_new_vehicles(self):
        for i in range(self.envdict['lane_count']):
            lane = self.lanes[i]
            density = 1000 * len(lane) / (self.envdict['length_backward'] + self.envdict['length_forward'])
            if density == 0:
                desdist = 0
            else:
                desdist = 1000.0 / self.envdict['density_lanes']
            if density < self.envdict['density_lanes']:
                # Need to create
                # random front-back
                r = np.random.rand() < 0.5
                if r == 0:
                    # back generation
                    vehiclecnt = len(lane)
                    if vehiclecnt == 0:
                        firstlength = 100000.0
                        vnext = 100000.0
                    else:
                        firstlength = lane[0].x - lane[0].length + self.envdict['length_backward']
                        vnext = lane[0].vx
                    if firstlength > desdist:
                        ev = Envvehicle(self.envdict)
                        """
                        ev.desired_speed = self.envdict['speed_lane' + str(i)][0] + np.random.randn() * \
                                                                                      self.envdict[
                                                                                          'speed_lane' + str(i)][0]
                        """
                        ev.desired_speed = self.envdict['speed_lane' + str(i)][0] + np.random.randn() * \
                                           ((self.envdict['speed_lane' + str(i)][1]) / 3.6)

                        ev.vx = min(ev.desired_speed, vnext)

                        ev.x = -self.envdict['length_backward'] + 10
                        ev.y = i * self.envdict['lane_width']
                        if i == 0:
                            ev.color = 'b'
                        else:
                            ev.color = 'k'
                        ev.laneindex = i
                        self.id = self.id + 1
                        ev.ID = self.id
                        lane.insert(0, ev)
                    else:
                        vehiclecnt = len(lane)
                        if vehiclecnt == 0:
                            firstlength = 100000.0
                            vnext = 100000.0
                        else:
                            last = lane[-1]
                            firstlength = abs(self.envdict['length_forward'] - last.x)
                            vnext = last.vx
                        if firstlength > desdist:
                            ev = Envvehicle(self.envdict)
                            """
                            ev.desired_speed = self.envdict['speed_lane' + str(i)][0] + np.random.randn() * \
                                                                                          self.envdict[
                                                                                              'speed_lane' + str(i)][1]
                            """
                            ev.desired_speed = self.envdict['speed_lane' + str(i)][0] + np.random.randn() * \
                                               ((self.envdict['speed_lane' + str(i)][1]) / 3.6)
                            ev.vx = min(ev.desired_speed, vnext)

                            ev.x = self.envdict['length_forward'] - 10
                            ev.y = i * self.envdict['lane_width']
                            if i == 0:
                                ev.color = 'b'
                            else:
                                ev.color = 'k'
                            ev.laneindex = i
                            self.id = self.id + 1
                            ev.ID = self.id
                            dist_front = abs(lane[len(lane) - 1].x - ev.x)
                            dist_left = 200

                            if i < self.envdict['lane_count'] - 1:
                                lane_left = self.lanes[i + 1]
                                veh_left = lane_left[len(lane_left) - 1]
                                dist_left = abs(veh_left.x - ev.x)

                                if (dist_front > 20) and (veh_left.state != 'switch_lane_right'):
                                    lane.insert(len(lane), ev)
                                    #print("                     Left is NOT switching lane")
                                elif (dist_front > 20) and (veh_left.state == 'switch_lane_right'):
                                    if (dist_left > 20):
                                        lane.insert(len(lane), ev)
                                        #print("                     Left is switching lane")
                                else:
                                    print("                     INSERT FORBIDDEN!!!")
                                    #print("first x: " + str(lane[len(lane) - 1].x) + " ev.x: " + str(ev.x))
                            else:
                                lane.insert(len(lane), ev)

    def random_new_des_speed(self):
        factor = 0.05
        for i in range(self.envdict['lane_count']):
            lane = self.lanes[i]
            for ev in lane:
                if (ev is Envvehicle) and (np.random.rand() < factor):
                    """
                    ev.desired_speed = self.envdict['speed_lane' + str(i)][0] + np.random.randn() * self.envdict[
                        'speed_lane' + str(i)][1]
                    """
                    ev.desired_speed = self.envdict['speed_lane' + str(i)][0] + np.random.randn() * \
                                       ((self.envdict['speed_lane' + str(i)][1]) / 3.6)

    def generate_state_for_ego(self):
        """
        Calculating the environment for the ego vehicle


        :return: np.array()
            idx   |  Meaning          |  Elements   |  Default
            ------+-------------------+-------------+----------
            0,1   | Front Left  Lane  |  dx,dv      |  500,0
            2,3   | Front Ego   Lane  |  dx,dv      |  500,0
            4,5   | Front Right Lane  |  dx,dv      |  500,0
            6,7   | Rear  Left  Lane  |  dx,dv      |  500,0
            8,9   | Rear  Ego   Lane  |  dx,dv      |  500,0
            10,11 | Rear  Right Lane  |  dx,dv      |  500,0
            12    | Left  Safe Zone   |  Occup [0,1]|  -
            13    | Right Safe Zone   |  Occup [0,1]|  -
            14    | Vehicle y pos     |  pos [m]    |  -
            15    | Vehicle heading   |  th[rad]    |  -
            16    | Vehicle speed     |  v[m/s]     |  -
        """
        state = np.zeros(17)
        state[0] = 500.
        state[2] = 500.
        state[4] = 500.
        state[6] = 500.
        state[8] = 500.
        state[10] = 500.

        lc = self.envdict['lane_count']
        # Left
        if not (self.egovehicle.laneindex == lc - 1):
            state[0], state[1], state[6], state[7], state[12] = self.searchlane_forstate(
                self.lanes[self.egovehicle.laneindex + 1], self.egovehicle.x)
        # right
        if self.egovehicle.laneindex != 0:
            state[4], state[5], state[10], state[11], state[13] = self.searchlane_forstate(
                self.lanes[self.egovehicle.laneindex - 1], self.egovehicle.x)
        # Ego lane
        state[2], state[3], state[8], state[9], _ = self.searchlane_forstate(self.lanes[self.egovehicle.laneindex],
                                                                             self.egovehicle.x)
        state[14] = round(self.egovehicle.y, 3)
        state[15] = round(math.atan2(self.egovehicle.vy, self.egovehicle.vx), 6)
        state[16] = round(math.sqrt(self.egovehicle.vx ** 2 + self.egovehicle.vy ** 2), 3)
        return state

    def searchlane_forstate(self, lane, x):
        szlen = 2  # safe zone length
        rear = None
        safe = None
        front = None
        res = np.array([500., 0., 500., 0. , 0. ])
        egorear = self.egovehicle.x - self.egovehicle.length - szlen
        egofront = self.egovehicle.x + szlen
        egox = self.egovehicle.x
        egov = self.egovehicle.vx
        for i in range(len(lane)):
            if lane[i].x < egorear:
                rear = lane[i]
            elif lane[i].x - lane[i].length < egofront:
                safe = lane[i]
            else:
                front = lane[i]
                break
        if not (front is None):
            res[0] = front.x - front.length - egox
            res[1] = front.vx - egov
        if not (rear is None):
            res[2] = egox - self.egovehicle.length - rear.x
            res[3] = egov - rear.vx
        if not (safe is None):
            res[4] = 1
        return res

    def searchEgoVehicle(self, preferredlaneid=-1):
        if preferredlaneid==-1:
            laneind= np.random.randint(0,self.envdict['lane_count'])
        else:
            laneind=preferredlaneid

        lane = self.lanes[laneind]
        ind = -1
        dist = 100000.0
        for i in range(len(lane)):
            if abs(lane[i].x) < dist:
                ind = i
                dist = abs(lane[i].x)
            else:
                break

        if ind == -1:
            e = Egovehicle(self.envdict)
            self.egovehicle = e
            self.lanes[laneind].append(e)
            e.x = 0
            e.y = 0
            e.vy = 0
            e.vx = e.desired_speed
            e.laneindex = laneind
            e.desired_speed=self.envdict['speed_ego_desired']
            return

        offs = -lane[ind].x
        for j in range(self.envdict['lane_count']):
            lane = self.lanes[j]
            for i in range(len(lane)):
                lane[i].x = lane[i].x + offs
        self.egovehicle = self.lanes[laneind][ind]
        for j in range(self.envdict['lane_count']):
            lane = self.lanes[j]
            for a in lane:
                if a.x <= -self.envdict['length_backward'] or a.x > self.envdict['length_forward']:
                    lane.remove(a)
        ind = self.lanes[laneind].index(self.egovehicle)
        old = self.egovehicle

        e = Egovehicle(self.envdict)
        self.egovehicle = e
        self.lanes[laneind][ind] = e
        e.x = old.x
        e.y = old.y
        e.vy = 0
        e.vx = old.vx
        e.laneindex = laneind

    def warmup(self, render=True):
        warmuptime = 30  # [secs]

        for i in range(self.envdict['lane_count']):
            self.nextvehicle.append(self.calcnextvehiclefollowing(i))


        for _ in range(int(warmuptime / self.envdict['dt'])):
            for i in range(self.envdict['lane_count']):
                lane = self.lanes[i]
                vehiclecnt = len(lane)
                if vehiclecnt == 0:
                    firstlength = 100000.0
                    vnext = 100000.0
                else:
                    firstlength = lane[0].x - lane[0].length + self.envdict['length_backward']
                    vnext = lane[0].vx

                if firstlength > self.nextvehicle[i]:
                    ev = Envvehicle(self.envdict)
                    """
                    ev.desired_speed = self.envdict['speed_lane' + str(i)][0] + np.random.randn() * self.envdict[
                        'speed_lane' + str(i)][1]
                    """
                    ev.desired_speed = self.envdict['speed_lane' + str(i)][0] + np.random.randn() * \
                                       ((self.envdict['speed_lane' + str(i)][1]) / 3.6)
                    #ev.desired_speed = self.envdict['speed_mean_lane' + str(i)] + np.random.randn() * self.envdict[
                    #    'speed_std_lane' + str(i)]
                    ev.vx = min(ev.desired_speed, vnext)

                    ev.x = -self.envdict['length_backward']
                    ev.y = i * self.envdict['lane_width']
                    if i == 0:
                        ev.color = 'b'
                    else:
                        ev.color = 'k'
                        ev.color = np.random.rand(3, )
                    self.id = self.id + 1
                    ev.ID = self.id
                    ev.laneindex = i
                    lane.insert(0, ev)
                    self.nextvehicle[i] = self.calcnextvehiclefollowing(i)

                vehiclecnt = len(lane)
                for j in range(vehiclecnt):
                    vehiclecnt = len(lane)
                    if j < vehiclecnt:
                        veh = lane[j]
                        #veh.laneindex = i
                        if isinstance(veh, Envvehicle):
                            if j + 1 < vehiclecnt:
                                vnext = lane[j + 1]
                            else:
                                vnext = None
                            veh.warmup_step(vnext)
                            #veh.step(vnext, None, None, None, None, None)
                            if veh.x > self.envdict['length_forward']:
                                lane.remove(veh)

            if render:
                self.render()

    def render(self, close=False, rewards=None):
        plt.axes().clear()
        for i in range(self.envdict['lane_count']):
            for j in range(len(self.lanes[i])):
                if isinstance(self.lanes[i][j], Egovehicle) or self.lanes[i][j].skip == 0:
                    self.lanes[i][j].render()

        lf = self.envdict['length_forward']
        lb = -self.envdict['length_backward']
        lw = self.envdict['lane_width']
        lc = self.envdict['lane_count']

        lines = plt.plot([lb, lf], [(lc - .5) * lw, (lc - .5) * lw], 'k')
        plt.setp(lines, linewidth=.5)
        lines = plt.plot([lb, lf], [-lw / 2, -lw / 2], 'k')
        plt.setp(lines, linewidth=.5)
        for i in range(lc - 1):
            lines = plt.plot([lb, lf], [(i + .5) * lw, (i + .5) * lw], 'k--')
            plt.setp(lines, linewidth=.5)
        plt.axis('equal')
        if close:
            plt.xlim([-100, 100])
        else:
            plt.xlim([-self.envdict['length_backward'] - 100, self.envdict['length_forward'] + 100])

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        if self.egovehicle is not None:
            th=math.atan2(self.egovehicle.vy, self.egovehicle.vx)*180/math.pi
            v = math.sqrt(self.egovehicle.vx ** 2 + self.egovehicle.vy ** 2)*3.6
            tstr='Speed: %4.2f [km/h]\nTheta:  %4.2f [deg]\nPos:  %4.2f [m]\nLane: %4d \n ' % (v,th,self.egovehicle.y,self.egovehicle.laneindex )
            plt.text(0.05,0.95,tstr,transform=plt.axes().transAxes,verticalalignment='top', bbox=props, fontsize=12,family='monospace')

        if not (rewards is None):
            tstr = 'Lane reward: %5.3f\n   y reward:  %5.3f\n   v reward:  %5.3f\n   c reward:  %5.3f\n ' % (rewards[0],rewards[1],rewards[2],rewards[3])
            plt.text(0.05, 0.35, tstr, transform=plt.axes().transAxes, verticalalignment='top', bbox=props, fontsize=14,
                     family='monospace')

        plt.show(False)
        plt.pause(0.001)

    def calcnextvehiclefollowing(self, lane):
        mean = 1000 / self.envdict['density_lanes']
        return max(10, mean + np.random.randn() * 20)

    """
    def save_log(self):
        if self.log_cnt > self.logs_in_file:
            log_file = open(r'D:\log_file.txt', 'w+')
            for i in range(self.logs_in_file):
                entry = self.log_list[i]
                log_file.write(entry)
            log_file.close()
    """


