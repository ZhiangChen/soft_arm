#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped as PS

def talker():
    pub1 = rospy.Publisher('Robot_1/pose', PS, queue_size=10)
    pub2 = rospy.Publisher('Robot_2/pose', PS, queue_size=10)
    pub3 = rospy.Publisher('Robot_3/pose', PS, queue_size=10)
    pub4 = rospy.Publisher('Robot_4/pose', PS, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    ps1 = PS()
    ps2 = PS()
    ps3 = PS()
    ps4 = PS()
    ps1.pose.position.x = 1.0
    ps1.pose.position.y = 1.1
    ps1.pose.position.z = 1.2
    ps1.pose.orientation.x = 0.4
    ps1.pose.orientation.y = 0.5
    ps1.pose.orientation.z = 0.6
    ps1.pose.orientation.w = 0.7
    ps1.header.frame_id = "raw"
    ps2.pose.position.x = 2.0
    ps2.pose.position.y = 2.1
    ps2.pose.position.z = 2.2
    ps2.pose.orientation.x = 0.4
    ps2.pose.orientation.y = 0.5
    ps2.pose.orientation.z = 0.6
    ps2.pose.orientation.w = 0.7
    ps2.header.frame_id = "raw"
    ps3.pose.position.x = 3.0
    ps3.pose.position.y = 3.1
    ps3.pose.position.z = 3.2
    ps3.pose.orientation.x = 0.4
    ps3.pose.orientation.y = 0.5
    ps3.pose.orientation.z = 0.6
    ps3.pose.orientation.w = 0.7
    ps3.header.frame_id = "raw"
    ps4.pose.position.x = 4.0
    ps4.pose.position.y = 4.1
    ps4.pose.position.z = 4.2
    ps4.pose.orientation.x = 0.4
    ps4.pose.orientation.y = 0.5
    ps4.pose.orientation.z = 0.6
    ps4.pose.orientation.w = 0.7
    ps4.header.frame_id = "raw"
    while not rospy.is_shutdown():
        rospy.loginfo('.')
        pub1.publish(ps1)
        pub2.publish(ps2)
        pub3.publish(ps3)
        pub4.publish(ps4)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass