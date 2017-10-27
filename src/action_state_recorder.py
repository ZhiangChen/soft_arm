#!/usr/bin/env python
"""
action_state_recorder.py
Zhiang Chen, Oct 15, 2017

MIT License
"""
import rospy
from geometry_msgs.msg import PoseArray as PA
from geometry_msgs.msg import Vector3
import numpy as np
import pickle

class ASRecorder(object):
    """
    ASRecorder is action-state recorder
    """
    def __init__(self):
        self.sub1 = rospy.Subscriber('robot_pose', PA, self.callback1, queue_size=1)
        self.sub2 = rospy.Subscriber('action', Vector3, self.callback2, queue_size=1)
        self.action_state = list()
        self.crt_state = list()

    def callback1(self, pa):
        n_px = [pa.poses[i].position.x for i in range(4)]
        n_py = [pa.poses[i].position.y for i in range(4)]
        n_pz = [pa.poses[i].position.z for i in range(4)]
        n_ox = [pa.poses[i].orientation.x for i in range(4)]
        n_oy = [pa.poses[i].orientation.y for i in range(4)]
        n_oz = [pa.poses[i].orientation.z for i in range(4)]
        n_ow = [pa.poses[i].orientation.w for i in range(4)]
        self.crt_state = np.array([n_px,n_py,n_pz,n_ox,n_oy,n_oz,n_ow]).transpose()

    def callback2(self, v3):
        if self.crt_state.shape[0] != 0:
            crt_action = np.array([v3.x, v3.y, v3.z])
            crt_action_state = {"action":crt_action,"state":self.crt_state}
            print '*'*40
            print crt_action_state
            self.action_state.append(crt_action_state)
            with open("action_state_data",'wb') as wfp:
                pickle.dump(self.action_state, wfp)



if __name__=='__main__':
	rospy.init_node('ASRecorder',anonymous=True)
	recorder = ASRecorder()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down ROS node ASRecorder")
