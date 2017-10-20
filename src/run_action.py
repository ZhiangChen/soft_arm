#!/usr/bin/env python
"""
run_action.py
Zhiang Chen, Oct 15, 2017

MIT License
"""
import sys
import rospy
from soft_arm.srv import *
from geometry_msgs.msg import Vector3
import numpy as np

control = Vector3()
control.x = 20
control.y = 20
control.z = 0

def run_action(control):
	rospy.wait_for_service('airpress_control',timeout=5)
	try:
		client = rospy.ServiceProxy('airpress_control', OneSeg)
		resp = client(control)
		return resp.status
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e	

if __name__ == "__main__":
	rospy.init_node('run_action', anonymous=True)
	pub = rospy.Publisher('action', Vector3, queue_size=10)
	if len(sys.argv) == 4:
		control.x = float(sys.argv[1])
		control.y = float(sys.argv[2])
		control.z = float(sys.argv[3])
		print run_action(control)

	elif len(sys.argv)==2:
		if sys.argv[1]=='all':
			for x in range(0,45,5):
				for y in range(0,45,5):
					for z in range(0,45,5):
						control.x = x
						control.y = y
						control.z = z
						print run_action(control)
						rospy.sleep(4.0)
						print "done"
						pub.publish(control)

	elif len(sys.argv) == 3:
		if (sys.argv[1] == 'random'):
			for _ in range(int(sys.argv[2])):
				x,y,z = np.random.randint(0,41,size=3)
				print x, y, z
				control.x = x
				control.y = y
				control.z = z
				print run_action(control)
				rospy.sleep(5.0)
				print "done"
				pub.publish(control)

	else:
		print "run default"
		print run_action(control)
