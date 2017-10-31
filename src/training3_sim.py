#!/usr/bin/env python
"""
Training on simulation
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
from simulator import Sim


MAX_EPISODES = 1000
MAX_EP_STEPS = 200
X_OFFSET = 0.0
Y_OFFSET = 0.0
Z_OFFSET = 0.0
S_DIM = 3
A_DIM = 3
A_BOUND = 10.0
GOT_GOAL = -0.05
TRAIN_POINT = 10000

class Trainer(object):
    def __init__(self):
        """ Initializing DDPG """
        self.sim = Sim()
        self.ddpg = DDPG(a_dim=A_DIM, s_dim=S_DIM, batch_size=1, memory_capacity=10000)
        self.ep_reward = 0.0
        self.current_ep = 0
        self.current_step = 0
        self.current_action = np.array([.0, .0, .0])
        self.done = True # if the episode is done
        self.var = 1.0
        print("Initialized DDPG")

        """ Setting communication"""
        self.pc = PC()
        self.pc.header.frame_id = 'world'
        self.pub = rospy.Publisher('normalized_state', PC, queue_size=10)
        """
        self.sub = rospy.Subscriber('robot_pose', PA, self.callback, queue_size=1)
        self.pub = rospy.Publisher('normalized_state', PC, queue_size=10)
        rospy.wait_for_service('airpress_control', timeout=5)
        self.target_PS = PS()
        self.action_V3 = Vector3()
        self.state = PA()
        self.updated = False # if s is updated
        self.got_callback1 = False
        self.got_callback2 = False
        print("Initialized communication")
        """

        """ Reading targets """
        """ The data should be w.r.t origin by base position """
        self.ends = pickle.load(open('./data/ends.p', 'rb'))
        self.x_offset = X_OFFSET
        self.y_offset = Y_OFFSET
        self.z_offset = Z_OFFSET
        self.scaler = 1/0.3
        self.sample_target()
        print("Read target data")

        #self.ddpg.restore_momery()
        #self.ddpg.restore_model()

        while not (rospy.is_shutdown()):
            if self.current_ep < MAX_EPISODES:
                if self.current_step < MAX_EP_STEPS:
                    #rospy.sleep(0.5)
                    p = self.sim.current_pose
                    s = np.vstack((p, self.target))
                    s = s[3:,:]
                    s = self.normalize_state(s)
                    norm_a = self.ddpg.choose_action(s.reshape(6)[-S_DIM:])
                    noise_a = np.random.normal(norm_a, self.var)
                    action = np.clip(noise_a, -1.0, 1.0)
                    self.current_action = action*A_BOUND + A_BOUND

                    p_ = self.sim.update_pose(self.current_action)
                    s_ = np.vstack((p_, self.target))

                    s_ = s_[3:,:]
                    s_ = self.normalize_state(s_)
                    self.compute_reward(s[0,:], s_[1,:])
                    self.ddpg.store_transition(s.reshape(6)[-S_DIM:], action, self.reward, s_.reshape(6)[-S_DIM:])

                    if self.ddpg.pointer > TRAIN_POINT:
                        #if (self.current_step % 10 == 0):
                        self.var *= 0.99999
                        self.ddpg.learn()

                    self.current_step += 1
                    self.ep_reward += self.reward

                    if self.reward > GOT_GOAL:
                        self.done = True
                        self.current_step = 0
                        self.current_ep += 1
                        self.sample_target()
                        #print "Target Reached"
                        if self.current_ep% 10 ==0:
                            print('\n')
                            print("Normalized action:")
                            print norm_a
                            print("Noise action:")
                            print action
                            print("Output action:")
                            print self.current_action
                            print("Reward: %f" % self.reward)
                            print('Episode:', self.current_ep, ' Reward: %f' % self.ep_reward, 'Explore: %.3f' % self.var,)
                            print('*' * 40)
                        self.ep_reward = 0
                        self.ddpg.save_memory()
                        #self.ddpg.save_model()
                        """
                        self.current_action = np.array([.0, .0, .0])
                        self.action_V3.x, self.action_V3.y, self.action_V3.z \
                            = self.current_action[0], self.current_action[1], self.current_action[2]
                        self.run_action(self.action_V3)
                        """

                    else:
                        self.done = False
                        if self.current_step == MAX_EP_STEPS:
                            #print "Target Failed"
                            if self.current_ep % 50 ==0:
                                print('\n')
                                print("Normalized action:")
                                print norm_a
                                print("Noise action:")
                                print action
                                print("Output action:")
                                print self.current_action
                                print("Reward: %f" % self.reward)
                                print('Episode:', self.current_ep, ' Reward: %f' % self.ep_reward, 'Explore: %.3f' % self.var,)
                                print('*' * 40)
                            self.current_step = 0
                            self.current_ep += 1
                            self.sample_target()
                            self.ep_reward = 0
                            self.ddpg.save_memory()
                            #self.ddpg.save_model()
                            """
                            self.current_action = np.array([.0, .0, .0])
                            self.action_V3.x, self.action_V3.y, self.action_V3.z \
                                = self.current_action[0], self.current_action[1], self.current_action[2]
                            self.run_action(self.action_V3)
                            """
                    self.pub_state(s_)
            else:
                p = self.sim.current_pose
                s = np.vstack((p, self.target))
                s = s[3:,:]
                s = self.normalize_state(s)
                norm_a = self.ddpg.choose_action(s.reshape(6)[-S_DIM:])
                self.current_action = norm_a * A_BOUND + A_BOUND
                print("Normalized action:")
                print norm_a
                print("Current action:")
                print self.current_action

                p_ = self.sim.update_pose(self.current_action)
                s_ = np.vstack((p_, self.target))
                s_ = s_[3:,:]
                s_ = self.normalize_state(s_)

                self.compute_reward(s_[0, :], s_[1, :])
                print('Explore: %.2f' % self.var,)
                rospy.sleep(1)
                #print
                print self.ddpg.get_value(s.reshape(6)[-S_DIM:],norm_a,self.reward.reshape((-1,1)),s_.reshape(6)[-S_DIM:])
                print '\n'
                self.pub_state(s_)

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
                    # self.ddpg.save_memory()
                    # self.ddpg.save_model()
                    """
                    self.current_action = np.array([.0, .0, .0])
                    self.action_V3.x, self.action_V3.y, self.action_V3.z \
                        = self.current_action[0], self.current_action[1], self.current_action[2]
                    self.run_action(self.action_V3)
                    """

                else:
                    self.done = False
                    self.current_step += 1
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
                        # self.ddpg.save_memory()
                        # self.ddpg.save_model()
                        """
                        self.current_action = np.array([.0, .0, .0])
                        self.action_V3.x, self.action_V3.y, self.action_V3.z \
                            = self.current_action[0], self.current_action[1], self.current_action[2]
                        self.run_action(self.action_V3)
                        """


    """
    def callback(self, pa):
        n_px = (np.array([pa.poses[i].position.x for i in range(4)]) - self.x_offset)*self.scaler
        n_py = (np.array([pa.poses[i].position.y for i in range(4)]) - self.y_offset)*self.scaler
        n_pz = (np.array([pa.poses[i].position.z for i in range(4)]) - self.z_offset)*self.scaler
        self.end = np.array((n_px[3], n_py[3], n_pz[3]))
        self.n_t = (self.target - np.array([0,0,Z_OFFSET]))*self.scaler
        self.s = np.concatenate((n_px, n_py, n_pz, self.n_t))
        self.pub_state(n_px, n_py, n_pz, self.n_t)
        self.updated = True
    """


    def sample_target(self):
            self.target = self.ends[np.random.randint(self.ends.shape[0])]
    """
    def run_action(self,control):
        try:
            client = rospy.ServiceProxy('airpress_control', OneSeg)
            resp = client(control)
            return resp.status
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    """

    def compute_reward(self,end,target):
        error = target - end
        self.reward = -np.linalg.norm(error)
        #print np.linalg.norm(error)

    """
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
    """
    def pub_state(self, state):
        pts = list()
        for i in range(state.shape[0]):
            pt = Point()
            pt.x = state[i,0]
            pt.y = state[i,1]
            pt.z = state[i,2]
            pts.append(pt)
        self.pc.points = pts
        self.pub.publish(self.pc)

    def normalize_state(self,state):
        offset = np.array([0,0,0.4])
        scaler = np.array([10,10,40])
        s = state - offset
        s = np.multiply(s, scaler)
        return s

if __name__ == '__main__':
    rospy.init_node('trainer',anonymous=True)
    trainer = Trainer()
    print("Shutting down ROS node trainer")