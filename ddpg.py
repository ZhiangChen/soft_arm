#!/usr/bin/env python
 
import sys
import rospy
from soft_arm.srv import *
from geometry_msgs.msg import Vector3

control = Vector3
control.x = 20
control.y = 20
control.z = 0

def display_vector(control):
	rospy.wait_for_service('normalized_control')
	try:
		client = rospy.ServiceProxy('normalized_control', OneSeg)
		resp = client(control)
		return resp.status
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e	

if __name__ == "__main__":
	if len(sys.argv) == 4:
		control.x = float(sys.argv[1])
		control.y = float(sys.argv[2])
		control.z = float(sys.argv[3])
	print display_vector(control)
