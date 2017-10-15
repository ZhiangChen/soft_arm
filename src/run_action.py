#!/usr/bin/env python
 
import sys
import rospy
from soft_arm.srv import *
from geometry_msgs.msg import Vector3

control = Vector3()
control.x = 20
control.y = 20
control.z = 0

def run_action(control):
	rospy.wait_for_service('airpress_control')
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

	elif sys.argv[1]=='all':
		for x in range(60):
			for y in range(60):
				for z in range(60):
					control.x = x
					control.y = y
					control.z = z
					print run_action(control)
					pub.publish(control)


	else:
		print run_action(control)
