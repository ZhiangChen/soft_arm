#!/usr/bin/env python
"""
Particle Filter Network
Zhiang Chen, Nov 24, 2017

MIT License
"""
import tensorflow as tf
import numpy as np
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

ALPHA = 0.01

class PFN(object):
    def __init__(self, lr=0.001, tau=0.00001, batch_size=64, a_dim=3, s_dim=31,
                 memory_capacity=10000, bound = 10):
        """
        :param lr_r: learning rate of actor
        :param tau: entropy parameter
        :param batch_size: batch size
        :param a_dim: action dimension
        :param s_dim: state dimension
        :param a_bound: bound scale of action
        :param memory_capacity: memory capacity
        """
        self.lr, self.tau, self.batch_size, self.memory_capacity, self.bound = \
            lr, tau, batch_size, memory_capacity, bound

        self.memory = np.zeros((self.memory_capacity, s_dim + a_dim + 1 + 1), dtype=np.float32)
        self.pointer = 0 # memory pointer
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.A = tf.placeholder(tf.float32, [None, a_dim], 'a')
        self.W = tf.placeholder(tf.float32, [None, 1], 'w')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        with tf.variable_scope('KFN'):
            self.a, self.sigma = self._build_net(self.S, scope='actor', trainable=True)
            normal_dist = tf.distributions.Normal(self.a, self.sigma + 1e-4)

        # networks parameters
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KFN/actor')

        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.params])
        self.l2_var = tf.norm(self.sigma)

        with tf.name_scope('choose_action'):  # use local params to choose action
            self.action = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0)*self.bound + self.bound, 0, 2*self.bound)
            self.prob = normal_dist.prob(self.action)
            self.mean_action = tf.clip_by_value(self.a*self.bound + self.bound, 0, 2*self.bound)

        # set optimizer
        with tf.name_scope('optimizer'):
            with tf.variable_scope('loss'):
                MSE = tf.losses.mean_squared_error(labels=self.A, predictions=self.a*self.bound + self.bound)
                entropy = normal_dist.entropy()
                loss = MSE - self.tau*(entropy - 0.5*self.l2_var)
            self.train = tf.train.AdamOptimizer(self.lr).minimize(loss, var_list=self.params)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        tf.summary.FileWriter("logs/", self.sess.graph)


    def learn(self):
        if self.pointer > self.memory_capacity:
            indices = np.random.choice(self.memory.shape[0], size=self.batch_size)
        else:
            indices = np.random.choice(self.pointer, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]

        self.sess.run(self.train, {self.S: bs, self.A:ba, self.is_training:True})

    def raw_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :], self.is_training: False})[0]


    def choose_action(self, s):
        a, p = self.sess.run([self.action, self.prob], {self.S: s[np.newaxis, :], self.is_training:False})
        return a[0], np.average(p[0])

    def choose_action2(self, s):
        a, p = self.sess.run([self.mean_action, self.sigma], {self.S: s[np.newaxis, :], self.is_training: False})
        return a[0], np.average(p[0])

    def store_transition(self, s, a, r, var_w):
        """
        Store transition
        :return: None
        """
        transition = np.hstack((s, a, [r], [var_w]))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1


    def save_memory(self):
        M = {"memory":self.memory, "pointer": self.pointer}
        with open("./data/memory.p", 'wb') as wfp:
            pickle.dump(M, wfp)
        #print("Saved memory")
        #print("Pointer location: " + str(self.pointer))

    def save_model(self, loc = 'model'):
        save_path = self.saver.save(self.sess, "./" + loc + "/model.ckpt")
        print("Model saved in file: %s" % save_path)

    def restore_model(self, loc = 'model'):
        print("Restored model")
        self.saver.restore(self.sess, "./" + loc + "/model.ckpt")

    def restore_momery(self):
        M = pickle.load(open('./data/memory.p', 'rb'))
        self.memory = M["memory"]
        self.pointer = M["pointer"]
        print("Restored memory")
        print("Pointer location: %i" % self.pointer)

    def _build_net(self, s, scope, trainable):
        hid_num1 = 200
        hid_num2 = 20
        with tf.variable_scope(scope):
            #bn1 = tf.layers.batch_normalization(s, axis=1, training=self.is_training, name='bn1', trainable=trainable)
            hidden1 = tf.layers.dense(s, hid_num1, activation=tf.nn.relu, name='fc1', trainable=trainable,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            #bn2 = tf.layers.batch_normalization(hidden1, axis=-1, training=self.is_training, name='bn2', trainable=trainable)
            hidden2 = tf.layers.dense(hidden1, hid_num2, activation=tf.nn.relu, name='fc2', trainable=trainable,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            #scaled_a = tf.layers.dense(hidden2, self.a_dim, activation=tf.nn.tanh, name='scaled_a', trainable=trainable,
            #                           kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
            scaled_a = tf.layers.dense(hidden2, self.a_dim, activation=None, name='scaled_a', trainable=trainable,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
            sigma = tf.layers.dense(hidden2, self.a_dim, tf.nn.softplus, name='sigma', trainable=trainable,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            return scaled_a, sigma

