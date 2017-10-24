#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud as PC
from geometry_msgs.msg import Point

def talker():
    pub1 = rospy.Publisher('state', PC, queue_size=10)
    rospy.init_node('state_pub', anonymous=True)
    rate = rospy.Rate(1)  # 50hz
    count = 0
    pc = PC()
    pc.header.frame_id = 'world'
    pts = list()

    for i in range(10):
        point = Point()
        point.x = 0
        point.y = 0
        point.z = i*0.1
        pts.append(point)
    print [x.z for x in pts]
    pc.points = pts

    while not rospy.is_shutdown():
        rospy.loginfo('Publishing state...')
        pub1.publish(pc)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
