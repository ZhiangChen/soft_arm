#!/usr/bin/env python
"""
Using data to simulate soft arm
Zhiang Chen, Oct 16, 2017

MIT License
"""
import pickle
import numpy as np
from sensor_msgs.msg import PointCloud as PC
import rospy
from geometry_msgs.msg import Point

class Sim(object):
    """
    - read data
    - update state by linear interpolation
    """
    def __init__(self):
        self.pub = rospy.Publisher('state_0', PC, queue_size=10)
        self.pc = PC()
        self.pc.header.frame_id = 'world'

        a_s = pickle.load(open('./data/action_state_data', 'rb'))
        as_array = np.zeros((len(a_s), 5, 3))
        for i, ans in enumerate(a_s):
            as_array[i, 4, :] = ans['action']
            as_array[i, :4, :] = ans['state'][:, :3]

        self.as_array_0 = np.zeros((len(a_s), 5, 3))
        for i, asn in enumerate(a_s):
            self.as_array_0[i, 4, :] = asn['action']
            self.as_array_0[i, :4, :] = asn['state'][:, :3] - asn['state'][0, :3]
            self.pub_state(self.as_array_0[i,:4,:])
            #print self.as_array_0[i,:4,:].shape
            #rospy.sleep(0.02)

        self.states = self.as_array_0[:,:4,:]
        self.actions = self.as_array_0[:,4,:]
        self.current_pose = self.states[0,:,:]

    def update_pose(self, action):
        upper_a, upper_s = self.get_upper_action(action)
        lower_a, lower_s = self.get_lower_action(action)
        d1 = upper_a - action
        d2 = action - lower_a
        if np.all(upper_a == lower_a):
            self.current_pose = upper_s
        else:
            d1 = np.sum(d1**2)
            d2 = np.sum(d2**2)
            w1 = d2/(d1+d2)
            w2 = d1/(d1+d2)
            self.current_pose = upper_s*w1 + lower_s*w2
        return self.current_pose


    def get_upper_action(self, action):
        delta_actions = self.actions - action
        x = delta_actions[:,0].tolist()
        y = delta_actions[:,1].tolist()
        z = delta_actions[:,2].tolist()
        ax = min(i for i in x if i >= 0)
        ay = min(i for i in y if i >= 0)
        az = min(i for i in z if i >= 0)
        a = action + np.array([ax, ay, az])
        index = np.where(np.all(self.actions==a,axis=1)==True)[0][0]
        return a, self.states[index]


    def get_lower_action(self, action):
        delta_actions = self.actions - action
        x = delta_actions[:,0].tolist()
        y = delta_actions[:,1].tolist()
        z = delta_actions[:,2].tolist()
        ax = max(i for i in x if i <= 0)
        ay = max(i for i in y if i <= 0)
        az = max(i for i in z if i <= 0)
        a = action + np.array([ax, ay, az])
        index = np.where(np.all(self.actions == a, axis=1) == True)[0][0]
        return a, self.states[index]

    def pub_state(self, pose):
        pts = list()
        for i in range(4):
            pt = Point()
            pt.x = pose[i,0]
            pt.y = pose[i,1]
            pt.z = pose[i,2]
            pts.append(pt)
        self.pc.points = pts
        self.pub.publish(self.pc)


if __name__ == '__main__':
    rospy.init_node('simulator', anonymous=True)
    try:
        sim = Sim()
        a = np.array([4, 10, 14])
        print a
        print sim.get_upper_action(a)[0]
        print sim.get_lower_action(a)[0]
        print sim.get_upper_action(a)[1]
        print sim.get_lower_action(a)[1]
        print sim.update_pose(a)
    except rospy.ROSInterruptException:
        pass
