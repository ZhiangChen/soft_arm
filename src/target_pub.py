#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped as PS

def talker():
    pub1 = rospy.Publisher('Robot_5/pose', PS, queue_size=10)
    rospy.init_node('target_pub', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    ps1 = PS()
    ps1.pose.position.x = 5.0
    ps1.pose.position.y = 5.1
    ps1.pose.position.z = 5.2
    ps1.pose.orientation.x = 0.4
    ps1.pose.orientation.y = 0.5
    ps1.pose.orientation.z = 0.6
    ps1.pose.orientation.w = 0.7
    ps1.header.frame_id = "raw"
    while not rospy.is_shutdown():
        rospy.loginfo('.')
        pub1.publish(ps1)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass