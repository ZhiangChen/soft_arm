#!/usr/bin/env python
"""
pfn tracking real target
"""

from pf_network import *
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
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)
MAX_EPISODES = 1
MAX_EP_STEPS = 200
X_OFFSET = 0.0917
Y_OFFSET = -0.4439
Z_OFFSET = 0.039
S_DIM = 3
A_DIM = 3
A_BOUND = 10.0
TRAIN_POINT = 2000
MEMORY_NUM = 2000


class Trainer(object):
    def __init__(self):
        """ Initializing DDPG """
        self.sim = Sim()
        self.pfn = PFN(a_dim=A_DIM, s_dim=S_DIM, batch_size=8,
                       memory_capacity=MEMORY_NUM, lr=0.001, bound=A_BOUND)
        self.ep_reward = 0.0
        self.current_action = np.array([.0, .0, .0])
        self.done = True # if the episode is done
        self.reward_record = list()
        self.ep_record = list()
        self.fig = plt.gcf()
        self.fig.show()
        self.fig.canvas.draw()
        print("Initialized PFN")

        """ Setting communication"""
        self.pc = PC()
        self.pc.header.frame_id = 'world'
        self.pub = rospy.Publisher('normalized_state', PC, queue_size=10)
        self.pub1 = rospy.Publisher('state', PC, queue_size=10)

        self.sub = rospy.Subscriber('Robot_1/pose', PS, self.callback, queue_size=1)
        rospy.wait_for_service('airpress_control', timeout=5)
        self.target_PS = PS()
        self.action_V3 = Vector3()
        self.updated = False  # if s is updated
        self.got_target = False
        print("Initialized communication")

        """ Reading targets """
        """ The data should be w.r.t origin by base position """
        self.ends = pickle.load(open('./data/targets.p', 'rb'))
        self.x_offset = X_OFFSET
        self.y_offset = Y_OFFSET
        self.z_offset = Z_OFFSET
        self.sample_target()
        print("Read target data")

        #self.pfn.restore_momery()
        self.pfn.restore_model('model_pfn')

        memory_ep = np.ones((MAX_EP_STEPS, 3 + 3 + 1 + 1)) * -100
        self.current_ep = 0
        self.current_step = 0
        while not (rospy.is_shutdown()):
            self.updated = False
            while (not self.updated) & (not rospy.is_shutdown()):
                rospy.sleep(0.1)
            real_target = self.real_target.copy()
            s = self.normalize_state(real_target)
            action, act_var = self.pfn.choose_action2(s)
            self.action_V3.x, self.action_V3.y, self.action_V3.z \
                = action[0], action[1], action[2]
            self.run_action(self.action_V3)
            print '\n'
            #rospy.sleep(1.0)

            '''
            if self.current_ep < MAX_EPISODES:
                if self.current_step < MAX_EP_STEPS:
                    #rospy.sleep(0.5)
                    s = self.normalize_state(self.target)
                    #print 'x'
                    #print s
                    action, prob = self.pfn.choose_action(s)
                    s_ = self.sim.update_pose(action)[-1,:]
                    self.compute_reward(self.target, s_)

                    #print action
                    #print self.pfn.raw_action(s)
                    #print self.target
                    #print s_
                    #print np.linalg.norm(self.target - s_)
                    #print self.reward
                    transition = np.hstack((self.target, action, self.reward, prob))
                    memory_ep[self.current_step] = transition
                    self.current_step += 1

                    state = np.array([s_, self.target])
                    self.pub_state(state)

                    if self.pfn.pointer > TRAIN_POINT:
                        self.pfn.learn()

                else:

                    best_i = np.argsort(memory_ep[:,-2])[-1]
                    best_action = memory_ep[best_i,3:6]
                    best_s_ = self.sim.update_pose(best_action)[-1, :]
                    best_distance = np.linalg.norm(self.target - best_s_)

                    """ #best action
                    print '*'
                    j = np.argsort(memory_ep[:,-2])[0]
                    print memory_ep[best_i, :]
                    print memory_ep[j, :]
                    print best_s_
                    print self.target
                    print best_action
                    print best_distance
                    """

                    self.current_step = 0
                    mean_action, act_var = self.pfn.choose_action2(s)
                    mean_s_ = self.sim.update_pose(mean_action)[-1, :]
                    mean_distance = np.linalg.norm(mean_s_ - self.target)

                    target_action,w = self.compute_weighted_action(memory_ep)
                    s_ = self.sim.update_pose(target_action)[-1,:]
                    target_distance = np.linalg.norm(s_ - self.target)
                    self.compute_reward(self.target, s_)

                    self.pfn.store_transition(s, target_action, mean_distance, np.var(w))
                    self.pfn.store_transition(s, best_action, best_distance, w[best_i])
                    self.current_ep += 1
                    self.sample_target()
                    memory_ep = np.ones((MAX_EP_STEPS, 3 + 3 + 1 + 1)) * -100

                    if self.current_ep% 10 ==0:
                        self.reward_record.append(mean_distance)
                        self.ep_record.append(self.current_ep)
                        plt.plot(self.ep_record, self.reward_record)
                        plt.ylim([0.0, 0.1])
                        self.fig.canvas.draw()
                        self.fig.savefig('learning.png')
                        print('\n')
                        #print s
                        print('Episode:', self.current_ep)
                        print("Mean Action:")
                        print mean_action
                        print("Mean Distance:")
                        print mean_distance
                        print("Action Variance:")
                        print act_var
                        print("Target Action:")
                        print target_action
                        print("Target Distance:")
                        print target_distance
                        print("Weights Variance:")
                        print np.var(w)
                        print('*' * 40)
                        self.pfn.save_model()
                        self.pfn.save_memory()


            else:
                s = self.normalize_state(self.target)
                action, act_var = self.pfn.choose_action2(s)
                s_ = self.sim.update_pose(action)[-1,:]
                self.compute_reward(self.target, s_)

                print("Mean Action:")
                print action
                print("Action Variance:")
                print act_var
                print("Distance: %f" % np.linalg.norm(self.target-s_))
                print("Reward: %f" % self.reward)
                print '\n'
                state = np.array([s_, self.target])
                self.pub_state(state)
                rospy.sleep(1)
                self.sample_target()
                
            '''


    def callback(self, ps):
        x = ps.pose.position.x - self.x_offset
        y = ps.pose.position.y - self.y_offset
        z = ps.pose.position.z - self.z_offset
        self.real_target = np.array([x,y,z])
        self.updated = True

    def run_action(self,control):
        try:
            print control
            client = rospy.ServiceProxy('airpress_control', OneSeg)
            resp = client(control)
            return resp.status
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


    def compute_weighted_action(self, memory_ep):
        sum = np.sum(memory_ep[:,-2]) + 1e-6
        try:
            w = memory_ep[:,-2]/sum
        except:
            print("Sum %f" % sum)
            print w
        target_action = np.average(memory_ep[:,S_DIM:S_DIM+A_DIM], axis=0, weights=w)
        return target_action, w

    def sample_target(self):
            self.target = self.ends[np.random.randint(self.ends.shape[0])]

    def compute_reward(self,end,target):
        error = target - end
        self.reward = 20**(-np.log2(2.5*np.linalg.norm(error)))
        #print np.linalg.norm(error)

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
        pts = list()
        for i in range(state.shape[0]):
            pt = Point()
            pt.x = state[i, 0] / 10.0
            pt.y = state[i, 1] / 10.0
            pt.z = state[i, 2] / 35.0 + 0.42
            pts.append(pt)
        self.pc.points = pts
        self.pub1.publish(self.pc)


    def normalize_state(self,state):
        offset = np.array([0,0,0.42])
        scaler = np.array([10,10,35])
        s = state - offset
        s = np.multiply(s, scaler)
        return s

    def calculate_dist(self, state):
        offset = np.array([0, 0, 0.42])
        scaler = np.array([10, 10, 35])
        s = np.multiply(state,1.0/scaler)
        s += offset
        return  np.linalg.norm(s)



if __name__ == '__main__':
    rospy.init_node('trainer',anonymous=True)
    trainer = Trainer()
    print("Shutting down ROS node trainer")