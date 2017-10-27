#!/usr/bin/env python

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
X_OFFSET = 0.1794
Y_OFFSET = -0.2747
Z_OFFSET = 0.0395
S_DIM = 15
A_DIM = 3
A_BOUND = 12
GOT_GOAL = -2
TRAIN_POINT = 200

class Trainer(object):
    def __init__(self):
        """ Initializing DDPG """
        self.ddpg = DDPG(a_dim=A_DIM, s_dim=S_DIM, batch_size=10, memory_capacity=500)
        self.ep_reward = 0.0
        self.current_ep = 0
        self.current_step = 0
        self.current_action = np.array([.0, .0, .0])
        self.done = True # if the episode is done
        self.var = 8.0
        print("Initialized DDPG")

        """ Setting communication"""
        self.sub = rospy.Subscriber('robot_pose', PA, self.callback, queue_size=1)
        self.pub = rospy.Publisher('normalized_state', PC, queue_size=10)
        rospy.wait_for_service('airpress_control', timeout=5)
        self.target_PS = PS()
        self.action_V3 = Vector3()
        self.pc = PC()
        self.pc.header.frame_id = 'world'
        self.state = PA()
        self.updated = False # if s is updated
        self.got_callback1 = False
        self.got_callback2 = False
        print("Initialized communication")

        """ Reading targets """
        """ The data should be w.r.t origin by base position """
        self.ends = pickle.load(open('./data/ends.p', 'rb'))
        self.x_offset = X_OFFSET
        self.y_offset = Y_OFFSET
        self.z_offset = Z_OFFSET
        self.scaler = 1/0.3
        self.sample_target()
        print("Read target data")

        self.ddpg.restore_momery()
        #self.ddpg.restore_model()


        while not (rospy.is_shutdown()):
            if self.current_ep < MAX_EPISODES:
                if self.current_step < MAX_EP_STEPS:
                    while (not self.updated) & (not rospy.is_shutdown()) :
                        rospy.sleep(0.1)
                    s = self.s.copy()

                    delta_a = self.ddpg.choose_action(s)*A_BOUND + A_BOUND
                    print("Raw action:")
                    print delta_a
                    delta_a = np.random.normal(delta_a,self.var)
                    print("Noise action: ")
                    print delta_a
                    self.current_action = delta_a
                    self.current_action = np.clip(self.current_action, 0, A_BOUND*2)
                    self.action_V3.x, self.action_V3.y, self.action_V3.z \
                        = self.current_action[0], self.current_action[1], self.current_action[2]
                    self.run_action(self.action_V3)
                    rospy.sleep(2.5)
                    print "Current action:"
                    print self.current_action

                    self.updated = False
                    while (not self.updated) & (not rospy.is_shutdown()):
                        rospy.sleep(0.1)
                    s_ = self.s.copy()
                    self.compute_reward(self.end, self.n_t)

                    action = (self.current_action - A_BOUND) / A_BOUND/2
                    self.ddpg.store_transition(s, action, self.reward, s_)
                    # print("Experience stored")

                    if self.ddpg.pointer > TRAIN_POINT:
                        if (self.current_step % 2 == 0):
                            self.var *= 0.992
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
                        print('Episode:', self.current_ep, ' Reward: %d' % self.ep_reward, 'Explore: %.2f' % self.var,)
                        print('*' * 40)
                        self.ep_reward = 0
                        self.ddpg.save_memory()
                        self.ddpg.save_model()
                        """
                        self.current_action = np.array([.0, .0, .0])
                        self.action_V3.x, self.action_V3.y, self.action_V3.z \
                            = self.current_action[0], self.current_action[1], self.current_action[2]
                        self.run_action(self.action_V3)
                        """

                    else:
                        self.done = False
                        if self.current_step == MAX_EP_STEPS:
                            print "Target Failed"
                            print("Episode %i Ends" % self.current_ep)
                            print(
                            'Episode:', self.current_ep, ' Reward: %d' % self.ep_reward, 'Explore: %.2f' % self.var,)
                            print('*' * 40)
                            self.current_step = 0
                            self.current_ep += 1
                            self.sample_target()
                            self.ep_reward = 0
                            self.ddpg.save_memory()
                            self.ddpg.save_model()
                            """
                            self.current_action = np.array([.0, .0, .0])
                            self.action_V3.x, self.action_V3.y, self.action_V3.z \
                                = self.current_action[0], self.current_action[1], self.current_action[2]
                            self.run_action(self.action_V3)
                            """
                    self.updated = False
                    print('\n')




    def callback(self, pa):
        n_px = (np.array([pa.poses[i].position.x for i in range(4)]) - self.x_offset)*self.scaler
        n_py = (np.array([pa.poses[i].position.y for i in range(4)]) - self.y_offset)*self.scaler
        n_pz = (np.array([pa.poses[i].position.z for i in range(4)]) - self.z_offset)*self.scaler
        self.end = np.array((n_px[3], n_py[3], n_pz[3]))
        self.n_t = self.target*self.scaler
        self.s = np.concatenate((n_px, n_py, n_pz, self.n_t))
        self.pub_state(n_px, n_py, n_pz, self.n_t)
        self.updated = True


    def sample_target(self):
            self.target = self.ends[np.random.randint(self.ends.shape[0])]
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
        self.pub.publish(self.pc)





if __name__ == '__main__':
    rospy.init_node('trainer',anonymous=True)
    trainer = Trainer()
    trainer.action_V3.x, trainer.action_V3.y, trainer.action_V3.z \
        = 0.0, 0.0, 0.0
    trainer.run_action(trainer.action_V3)
    print("Shutting down ROS node trainer")