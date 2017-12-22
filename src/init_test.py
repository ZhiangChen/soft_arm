import rospy
import numpy as np
from geometry_msgs.msg import PoseArray as PA
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped as PS
from soft_arm.srv import *
from sensor_msgs.msg import PointCloud as PC
from geometry_msgs.msg import Point
import pickle
from simulator import Sim
import matplotlib.pyplot as plt

class IniTest(object):
    def __init__(self):
        self.sub1 = rospy.Subscriber('Robot_1/pose', PS, self.callback1, queue_size=1)
        self.sub2 = rospy.Subscriber('Robot_2/pose', PS, self.callback2, queue_size=1)
        while not (rospy.is_shutdown()):
            self.updated1 = False
            self.updated2 = False
            while (not self.updated1) & (not self.updated2) & (not rospy.is_shutdown()):
                rospy.sleep(0.1)
            self.p2 = np.array([0.0917, -0.4439, 0.0390])
            print "Origin:" + str(self.p2)
            print "End_0:" + str(self.p1 - self.p2)
            print '\n'
            rospy.sleep(1.0)

    def callback1(self, ps):
        x = ps.pose.position.x
        y = ps.pose.position.y
        z = ps.pose.position.z
        self.p1 = np.array([x, y, z])
        self.updated1 = True

    def callback2(self, ps):
        x = ps.pose.position.x
        y = ps.pose.position.y
        z = ps.pose.position.z
        self.p2 = np.array([x, y, z])
        self.updated2 = True


if __name__ == '__main__':
    rospy.init_node('tester',anonymous=True)
    trainer = IniTest()
    print("Shutting down ROS node trainer")