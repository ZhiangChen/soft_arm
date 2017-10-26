import rospy
from std_msgs.msg import String

class Test(object):
    def __init__(self):
        self.sub1 = rospy.Subscriber('robot_pose', String, self.callback1, queue_size=1)
        self.s = ''
        while not (rospy.is_shutdown()):
            print self.s
            rospy.sleep(1)


    def callback1(self, s):
        self.s = s

if __name__ == '__main__':
    rospy.init_node('tester',anonymous=True)
    trainer = Test()
    print("Shuted down ROS node tester")