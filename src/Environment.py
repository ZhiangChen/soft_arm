#!/usr/bin/env python2
"""
Environment
Zhiang Chen, Oct 14, 2017

MIT License
"""

import rospy
from geometry_msgs.msg import PoseStamped as PS

class Env(object):
    """
    Env receives states.
    ***State:***
    (1) Static control:
        - poses of rigid bodies on the soft arm
        - target position
    (2) Dynamic control:
        - 3 timesteps poses of rigid bodies on the soft arm
        - target position
    """
    def __init__(self, normalizer={'x':0,'y':0,'z':0,'s':1} ):
        """
        - normalizer = {'x':x,'y':y,'z':z,'s'=s}
        - subscribers : 'Robot_1/pose', ..., 'Robot_4/pose'
        """
        self.x = normalizer['x']
        self.y = normalizer['y']
        self.z = normalizer['z']
        self.s = normalizer['s']
        sub1 = rospy.Subscriber('Robot_1/pose', PS, self.callback1, queue_size=1)
        sub2 = rospy.Subscriber('Robot_2/pose', PS, self.callback2, queue_size=1)
        sub3 = rospy.Subscriber('Robot_3/pose', PS, self.callback3, queue_size=1)
        sub4 = rospy.Subscriber('Robot_4/pose', PS, self.callback4, queue_size=1)
        rospy.init_node('Environment', anonymous = True)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down ROS node Environment")

    def naive_filter(self):
        pass

    def read_static_pose(self):
        pass

    def read_dynamic_pose(self):
        pass

    def record_pose(self):
        pass

    def callback1(self):
        pass

    def callback2(self):
        pass

    def callback3(self):
        pass

    def callback4(self):
        pass

if __name__ == '__main__':
	env = Env()