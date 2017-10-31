#!/usr/bin/env python
"""
DDPG
Zhiang Chen, Oct 15, 2017

MIT License
"""
import tensorflow as tf
import numpy as np
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class DDPG(object):
    """
    - set hyper-parameters
    - build actor nets
    - build target nets
    - build optimization
    - build memory
    - learn
    - choose action
    - store transition
    """
    def __init__(self, lr_a=0.01, lr_c=0.001, gamma=0.99, tau=0.001, batch_size=16, a_dim=3, s_dim=31,
                 memory_capacity=10000):
        """
        :param lr_a: learning rate of actor
        :param lr_c: learning rate of critic
        :param gamma: reward discount
        :param tau: rate of soft replacement
        :param batch_size: batch size
        :param a_dim: action dimension
        :param s_dim: state dimension
        :param a_bound: bound scale of action
        :param memory_capacity: memory capacity
        """
        self.lr_a, self.lr_c, self.gamma, self.tau, self.batch_size, self.memory_capacity = \
            lr_a, lr_c, gamma, tau, batch_size, memory_capacity

        self.memory = np.zeros((self.memory_capacity, s_dim*2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0 # memory pointer
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - self.tau) * ta + self.tau * ea), tf.assign(tc, (1 - self.tau) * tc + self.tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        # set optimizer
        with tf.name_scope('Optimizer'):
            with tf.variable_scope('loss'):
                q_target = self.R + self.gamma * q_
                td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
                a_loss = - tf.reduce_mean(q)  # maximize the q
            self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(td_error, var_list=self.ce_params)
            self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(a_loss, var_list=self.ae_params)
        self.q_e = q
        self.q_t = q_target
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        tf.summary.FileWriter("logs/", self.sess.graph)

    def learn(self):
        """
        having soft replacement first, getting training batch from memory and training actor and critic
        :return: None
        """
        # soft target replacement
        self.sess.run(self.soft_replace)

        if self.pointer > self.memory_capacity:
            indices = np.random.choice(self.memory.shape[0], size=self.batch_size)
        else:
            indices = np.random.choice(self.pointer, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs, self.is_training:True})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.is_training:True})


    def choose_action(self, s):
        """
        choose an action from the actor
        :param s: current state, ndarray, shape = 31
        :return: current action from the actor
        """
        return self.sess.run(self.a, {self.S: s[np.newaxis, :], self.is_training:False})[0]

    def get_value(self, s, a, r, s_):
        return self.sess.run((self.q_t, self.q_e), {self.S: s[np.newaxis, :], self.a: a[np.newaxis, :],
                                                  self.R: r, self.S_: s_[np.newaxis, :], self.is_training: False})


    def store_transition(self, s, a, r, s_):
        """
        Store transition
        :param s: current state, ndarray, shape = 31 (3+4*7)
        :param a: current action, ndarray, shape = 3
        :param r: current reward, double, shape = 1
        :param s_: next state, ndarray, shape = 31
        :return: None
        """
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1


    def save_memory(self):
        M = {"memory":self.memory, "pointer": self.pointer}
        with open("./data/memory.p", 'wb') as wfp:
            pickle.dump(M, wfp)
        print("Saved memory")
        print("Pointer location: " + str(self.pointer))

    def save_model(self):
        save_path = self.saver.save(self.sess, "./model/model.ckpt")
        print("Model saved in file: %s" % save_path)

    def restore_model(self):
        print("Restored model")
        self.saver.restore(self.sess, "./model/model.ckpt")

    def restore_momery(self):
        M = pickle.load(open('./data/memory.p', 'rb'))
        self.memory = M["memory"]
        self.pointer = M["pointer"]
        print("Restored memory")
        print("Pointer location: %i" % self.pointer)

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            #bn1 = tf.layers.batch_normalization(s, axis=1, training=self.is_training, name='bn1', trainable=trainable)
            hidden1 = tf.layers.dense(s, 50, activation=tf.nn.sigmoid, name='fc1', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            #bn2 = tf.layers.batch_normalization(hidden1, axis=-1, training=self.is_training, name='bn2', trainable=trainable)
            hidden2 = tf.layers.dense(hidden1, 5, activation=tf.sigmoid, name='fc2', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            scaled_a = tf.layers.dense(hidden2, self.a_dim, activation=tf.nn.tanh, name='scaled_a', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            return scaled_a


    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            concat = tf.concat([s, a], axis=1, name='concat')
            #bn1 = tf.layers.batch_normalization(concat, axis=-1, training=self.is_training, name='bn1', trainable=trainable)
            hidden1 = tf.layers.dense(concat, 50, activation=tf.nn.relu, name='fc1', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            #bn2 = tf.layers.batch_normalization(hidden1, axis=-1, training=self.is_training, name='bn2', trainable=trainable)
            hidden2 = tf.layers.dense(hidden1, 5, activation=tf.nn.relu, name='fc2', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            q = tf.layers.dense(hidden1, 1, name='q', trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer())
            return q