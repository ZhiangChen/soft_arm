#!/usr/bin/env python
"""
- Receiving state geometry_msgs/PoseArray from 'agent state'
- Publishing target
- Training DDPG
- Publishing action
"""

from ddpg import *
import rospy
import numpy as np
from geometry_msgs.msg import PoseArray as PA
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped as PS
from soft_arm.srv import *
import pickle


MAX_EPISODES = 200
MAX_EP_STEPS = 200

np.random.seed(1)
tf.set_random_seed(1)


S_DIM = 15
A_DIM = 3
A_BOUND = 5
GOT_GOAL = -0.1

class Trainer():
    def __init__(self):
        """ Initializing DDPG """
        self.ddpg = DDPG(a_dim=A_DIM, a_bound=A_BOUND, s_dim=S_DIM)
        self.ep_reward = 0
        self.current_ep = 0
        self.current_step = 0
        self.current_action = np.array([.0, .0, .0])
        self.done = True
        self.var = 3.0
        self.updated = False

        """ Setting communication"""
        self.sub1 = rospy.Subscriber('agent_state', PA, self.callback1, queue_size=1)
        self.sub2 = rospy.Subscriber('agent_state', PA, self.callback1, queue_size=1)
        self.pub1 = rospy.Publisher('Robot_5/pose', PS, queue_size=10)
        rospy.wait_for_service('airpress_control', timeout=5)
        self.target_PS = PS()
        self.action_V3 = Vector3()

        """ Reading targets """
        aoe = pickle.load(open('action_origin_end.p', 'rb'))
        r_data = aoe['random']
        rs_data = aoe['random_step']
        self.r_ends = r_data[:, 2, :]
        self.rs_ends = rs_data[:, 2, :]

        """Initializing target"""
        self.pub1.publish(self.sample_target())



    def sample_target(self, e=0.6):
        if np.random.rand(1)[0] < e:
            self.target = self.r_ends[np.random.randint(self.r_ends.shape[0])]
            self.target_PS.pose.position.x, self.target_PS.pose.position.y, self.target_PS.pose.position.z\
                = self.target[0], self.target[1], self.target[2]
            return self.target_PS
        else:
            target = self.rs_ends[np.random.randint(self.rs_ends.shape[0])]
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

    def compute_reward(self):
        end = np.array((self.s_[3], self.s_[8], self.s_[13]))
        target = np.array((self.s_[4], self.s_[9], self.s_[14]))
        error = target - end
        self.reward = -np.linalg.norm(error)*10

    def callback1(self, pa):
        n_px = [pa.poses[i].position.x for i in range(5)]
        n_py = [pa.poses[i].position.y for i in range(5)]
        n_pz = [pa.poses[i].position.z for i in range(5)]
        self.s = np.array(n_px + n_py + n_pz)
        if self.current_ep < MAX_EPISODES:
            if self.current_step < MAX_EP_STEPS:
                delta_a = self.ddpg.choose_action(s)
                delta_a = np.random.normal(delta_a,self.var)
                self.current_action += delta_a
                self.current_action = np.clip(self.current_action, 0, 40)
                self.action_V3.x, self.action_V3.y, self.action_V3.z \
                    = self.current_action.x, self.current_action.y, self.current_action.z
                self.run_action(self.action_V3)
                rospy.sleep(4.0)
                print self.current_action
                self.updated = True
                if self.ddpg.pointer > ddpg.memory_capacity:
                    self.var *= 0.9999
                    self.ddpg.learn()

                self.current_step += 1

                if self.reward > GOT_GOAL:
                    self.done = True
                    self.current_step = 0
                    self.current_ep += 1
                    print "Reach Target"

                else:
                    self.done = False

                if self.current_step == MAX_EP_STEPS:
                    self.current_step = 0
                    self.current_ep += 1




    def callback2(self, pa):
        n_px = [pa.poses[i].position.x for i in range(5)]
        n_py = [pa.poses[i].position.y for i in range(5)]
        n_pz = [pa.poses[i].position.z for i in range(5)]
        self.s_ = np.array(n_px + n_py + n_pz)
        if self.updated:
            self.updated = False
            self.compute_reward()
            self.ddpg.store_transition(self.s, self.current_action, self.reward, self.s_)








var = 3.0  # control exploration
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0

    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > ddpg.memory_capacity:
            var *= .9999    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:RENDER = True
            break