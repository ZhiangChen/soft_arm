#!/usr/bin/env python
"""
- Receiving state geometry_msgs/PoseArray from 'agent state'
- Publishing target
- Training DDPG
- Publishing action
"""
""" Training
--- ROS
1. roslaunch mocap_optitrack mocap.launch
2. roslaunch soft_arm read_state
3. python training.py
--- Labview

"""

from ddpg import *
import rospy
import numpy as np
from geometry_msgs.msg import PoseArray as PA
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped as PS
from soft_arm.srv import *
from sensor_msgs.msg import PointCloud as PC
from geometry_msgs.msg import Point
import pickle


MAX_EPISODES = 200
MAX_EP_STEPS = 10
X_OFFSET = 0.1953
Y_OFFSET = -0.2834
Z_OFFSET = 0.0376
S_DIM = 15
A_DIM = 3
A_BOUND = 5
GOT_GOAL = -2
TRAIN_POINT = 500


np.random.seed(1)
tf.set_random_seed(1)




class Trainer():
    def __init__(self):
        """ Initializing DDPG """
        self.ddpg = DDPG(a_dim=A_DIM, s_dim=S_DIM, memory_capacity=10000)
        self.ep_reward = 0.0
        self.current_ep = 0
        self.current_step = 0
        self.current_action = np.array([.0, .0, .0])
        self.done = True # if the episode is done
        self.var = 3.0
        print("Initialized DDPG")

        """ Setting communication"""
        self.sub1 = rospy.Subscriber('robot_pose', PA, self.callback1, queue_size=1)
        self.sub2 = rospy.Subscriber('robot_pose', PA, self.callback2, queue_size=1)
        self.pub1 = rospy.Publisher('Robot_5/pose', PS, queue_size=10)
        self.pub2 = rospy.Publisher('normalized_state', PC, queue_size=10)
        rospy.wait_for_service('airpress_control', timeout=5)
        self.target_PS = PS()
        self.action_V3 = Vector3()
        self.pc = PC()
        self.pc.header.frame_id = 'world'
        self.state = PA()
        self.updated = False # if s_ is updated
        self.stored = True # if experience is stored and reward is computed
        self.got_callback1 = False
        self.got_callback2 = False
        print("Initialized communication")

        """ Reading targets """
        """ The data should be w.r.t origin by base position """
        ends = pickle.load(open('ends.p', 'rb'))
        self.r_ends = ends['r_ends']
        self.rs_ends = ends['rs_ends']
        self.x_offset = X_OFFSET
        self.y_offset = Y_OFFSET
        self.z_offset = Z_OFFSET
        self.scaler = 1/0.3
        self.sample_target()
        print("Read target data")

        self.ddpg.restore_momery()
        self.ddpg.restore_model()

        """
        while not (rospy.is_shutdown()):
            self.pub1.publish(self.target_PS)
            #print('pub')
            rospy.sleep(0.1)
        """


    def sample_target(self, e=0.2):
        if np.random.rand(1)[0] < e:
            self.target = self.r_ends[np.random.randint(self.r_ends.shape[0])]
            self.target_PS.pose.position.x, self.target_PS.pose.position.y, self.target_PS.pose.position.z\
                = self.target[0], self.target[1], self.target[2]
            return self.target_PS
        else:
            self.target = self.rs_ends[np.random.randint(self.rs_ends.shape[0])]
            self.target_PS.pose.position.x, self.target_PS.pose.position.y, self.target_PS.pose.position.z \
                = self.target[0], self.target[1], self.target[2]
            return self.target_PS

    def run_action(self,control):
        try:
            client = rospy.ServiceProxy('airpress_control', OneSeg)
            resp = client(control)
            return resp.status
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def compute_reward(self,end,target):
        error = target - end
        self.reward = -np.exp(np.linalg.norm(error)*10)
        print np.linalg.norm(error)
        print("Reward: %d" % self.reward)

    def pub_state(self, n_px, n_py, n_pz, n_t):
        pts = list()
        for i in range(4):
            pt = Point()
            pt.x = n_px[i]
            pt.y = n_py[i]
            pt.z = n_pz[i]
            pts.append(pt)
        pt = Point()
        pt.x, pt.y, pt.z = n_t[0], n_t[1], n_t[2]
        pts.append(pt)
        self.pc.points = pts
        self.pub2.publish(self.pc)


    def callback1(self, pa):
        n_px = (np.array([pa.poses[i].position.x for i in range(4)]) - self.x_offset)*self.scaler
        n_py = (np.array([pa.poses[i].position.y for i in range(4)]) - self.y_offset)*self.scaler
        n_pz = (np.array([pa.poses[i].position.z for i in range(4)]) - self.z_offset)*self.scaler
        n_t = self.target*self.scaler
        self.s = np.concatenate((n_px, n_py, n_pz, n_t))
        if self.current_ep < MAX_EPISODES:
            if self.current_step < MAX_EP_STEPS:
                if self.stored:
                    delta_a = self.ddpg.choose_action(self.s)*A_BOUND
                    print("Delta action:")
                    print delta_a
                    delta_a = np.random.normal(delta_a,self.var)
                    print("Noise delta: ")
                    print delta_a
                    self.current_action += delta_a
                    self.current_action = np.clip(self.current_action, 0, 30)
                    self.action_V3.x, self.action_V3.y, self.action_V3.z \
                        = self.current_action[0], self.current_action[1], self.current_action[2]
                    self.run_action(self.action_V3)
                    rospy.sleep(2.5)
                    print "Current action:"
                    print self.current_action
                    self.updated = True
                    self.stored = False



    def callback2(self, pa):
        #print 'having c2'
        n_px = (np.array([pa.poses[i].position.x for i in range(4)]) - self.x_offset) * self.scaler
        n_py = (np.array([pa.poses[i].position.y for i in range(4)]) - self.y_offset) * self.scaler
        n_pz = (np.array([pa.poses[i].position.z for i in range(4)]) - self.z_offset) * self.scaler
        end = np.array((n_px[3], n_py[3], n_pz[3]))
        n_t = self.target * self.scaler
        self.s_ = np.concatenate((n_px, n_py, n_pz, n_t))
        print("Current: ")
        print n_px[3], n_py[3], n_pz[3]
        print("Target: ")
        print n_t
        self.pub_state(n_px, n_py, n_pz, n_t)
        if self.updated:
            self.updated = False
            self.compute_reward(end, n_t)
            self.ddpg.store_transition(self.s, self.current_action, self.reward, self.s_)
            #print("Experience stored")

            if self.ddpg.pointer > TRAIN_POINT:
                self.var *= 0.9999
                self.ddpg.learn()

            self.current_step += 1
            self.ep_reward += self.reward

            if self.reward > GOT_GOAL:
                self.done = True
                self.current_step = 0
                self.current_ep += 1
                self.sample_target()
                print "Target Reached"
                print("Episode %i Ended" % self.current_ep)
                print('Episode:', self.current_ep, ' Reward: %i' % int(self.ep_reward), 'Explore: %.2f' % self.var,)
                print('*'*40)
                self.ep_reward = 0
                self.ddpg.save_memory()
                self.ddpg.save_model()
                self.current_action = np.array([.0, .0, .0])
                self.action_V3.x, self.action_V3.y, self.action_V3.z \
                    = self.current_action[0], self.current_action[1], self.current_action[2]
                self.run_action(self.action_V3)
                rospy.sleep(2.5)

            else:
                self.done = False
                if self.current_step == MAX_EP_STEPS:
                    print "Target Failed"
                    print("Episode %i Ends" % self.current_ep)
                    print('Episode:', self.current_ep, ' Reward: %f' % int(self.ep_reward), 'Explore: %.2f' % self.var,)
                    print('*' * 40)
                    self.current_step = 0
                    self.current_ep += 1
                    self.sample_target()
                    self.ep_reward = 0
                    self.ddpg.save_memory()
                    self.ddpg.save_model()
                    self.current_action = np.array([.0, .0, .0])
                    self.action_V3.x, self.action_V3.y, self.action_V3.z \
                        = self.current_action[0], self.current_action[1], self.current_action[2]
                    self.run_action(self.action_V3)
                    rospy.sleep(2.5)

            self.stored = True
            print('\n')


if __name__ == '__main__':
    rospy.init_node('trainer',anonymous=True)
    trainer = Trainer()
try:
    print "spin"
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down ROS node trainer")