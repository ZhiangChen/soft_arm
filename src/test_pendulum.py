#!/usr/bin/env python
"""
DDPG test with pendulum
Zhiang Chen, Oct 16, 2017

MIT License
"""

import gym
from ddpg import *

MAX_EPISODES = 200
MAX_EP_STEPS = 200
ENV_NAME = 'Pendulum-v0'
RENDER = False

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
np.random.seed(1)
tf.set_random_seed(1)


s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim=a_dim, s_dim=s_dim)
#var = 2.0  # control exploration
var = 0.5
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    #if i == 100:
    #   ddpg.save_model()
    #    ddpg.save_memory()
    #ddpg.restore_momery()
    #ddpg.restore_model()

    for j in range(MAX_EP_STEPS):
        if RENDER & (i%5==0):
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)*a_bound
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