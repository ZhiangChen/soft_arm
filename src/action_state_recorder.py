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

class ASRecorder(object):
    """
    ASRecorder is action-state recorder
    """
    def __init__(self):
        self.sub1 = rospy.Subscriber('agent_state', PA, self.callback1, queue_size=1)
        self.sub2 = rospy.Subscriber('action', Vector3, self.callback2, queue_size=1)
        self.action_state = list()
        self.crt_state = list()
        self.f = open("action_state_data", "w")

    def callback1(self, pa):
        n_px = [pa.poses[i].position.x for i in range(5)]
        n_py = [pa.poses[i].position.y for i in range(5)]
        n_pz = [pa.poses[i].position.z for i in range(5)]
        n_ox = [pa.poses[i].orientation.x for i in range(5)]
        n_oy = [pa.poses[i].orientation.y for i in range(5)]
        n_oz = [pa.poses[i].orientation.z for i in range(5)]
        n_ow = [pa.poses[i].orientation.w for i in range(5)]
        self.crt_state = np.array([n_px,n_py,n_pz,n_ox,n_oy,n_oz,n_ow]).transpose()

    def callback2(self, v3):
        if self.crt_state.len() != 0:
            crt_action = np.array([v3.x, v3.y, v3.z])
            crt_action_state = {"action":crt_action,"state":self.crt_state}
            print '*'*40
            print crt_action_state
            self.f.write(crt_action_state)

    def close_file(self):
        self.f.close()
        print "closed file"


if __name__=='__main__':
	rospy.init_node('ASRecorder',anonymous=True)
	recorder = ASRecorder()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down ROS node ASRecorder")