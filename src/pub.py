#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
control = Vector3()
control.x = 1.1
control.y = 2.21
control.z = 3.321

def talker():
	pub = rospy.Publisher('vector', Vector3, queue_size=10)
	pub1 = rospy.Publisher('chatter', String, queue_size=10)
	rospy.init_node('talker', anonymous=True)
	rate = rospy.Rate(10) # 10hz
	rospy.loginfo('.')
	while not rospy.is_shutdown():
		hello_str = "hello world %s" % rospy.get_time()
		rospy.loginfo(hello_str)
		pub.publish(control)		
		pub1.publish(hello_str)
		rate.sleep()
 
if __name__ == '__main__':
	try:
		talker()
	except rospy.ROSInterruptException:
		pass
