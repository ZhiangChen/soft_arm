#!/usr/bin/env python
"""
generate data by simulation
Zhiang Chen, Oct 30, 2017

MIT License
"""
import pickle
import numpy as np
import rospy
from simulator import Sim

rospy.init_node('simulator', anonymous=True)
sim = Sim()
actions = list()
poses = list()
for x in range(21):
    for y in range(21):
        for z in range(21):
            a = np.array([x,y,z]).astype(np.float32)
            p = sim.update_pose(a)
            actions.append(a)
            poses.append(p)

sim_data = {'actions':actions,'poses':poses}
pickle.dump( sim_data, open( "./data/sim_data.p", "wb" ) )
