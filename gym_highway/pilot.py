#https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import gym_highway
import os
from timeit import default_timer as timer

import csv

env = gym.make('EPHighWay-v0')

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden1 = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden2 = slim.fully_connected(hidden1, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden2, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


tf.reset_default_graph()  # Clear the Tensorflow graph.

myAgent = agent(lr=1e-4, s_size=17, a_size=3, h_size=512)  # Load the agent.

total_episodes = 150000  # Set total number of episodes to train agent on.
update_frequency = 5
gamma = 0.95
TRAIN=True
#TRAIN=False

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Launch the tensorflow graph
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
with tf.Session() as sess:
    sess.run(init)
    current_episode = 0
    total_reward = []
    total_length = []
    elapsed = 0
    mean_return_over_multiple_episodes = np.zeros(500)
    num_episodes_where_step_has_been_seen = np.zeros(500)

    if TRAIN:
        gradBuffer = sess.run(tf.trainable_variables())
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        while current_episode < total_episodes:
            s = env.reset()
            s = (s - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            running_reward = 0
            ep_history = []
            while True:
                start = timer()
                # Probabilistically pick an action given our network outputs.
                a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)
                s1, r, d, _ = env.step(a)
                s1 = (s1 - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
                ep_history.append([s, a, r, s1])
                s = s1
                running_reward += r
                if d == True:
                    # Update the network.
                    ep_history = np.array(ep_history)
                    # ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                    discounted_epr = discount_rewards(ep_history[:, 2])
                    for i in range(discounted_epr.shape[0]):
                        num_episodes_where_step_has_been_seen[i] += 1
                        mean_return_over_multiple_episodes[i] -= mean_return_over_multiple_episodes[i] / \
                                                                  num_episodes_where_step_has_been_seen[i]
                        mean_return_over_multiple_episodes[i] += discounted_epr[i] / \
                                                                  num_episodes_where_step_has_been_seen[i]
                        discounted_epr[i] -= mean_return_over_multiple_episodes[i]
                    feed_dict = {myAgent.reward_holder: discounted_epr,
                                 myAgent.action_holder: ep_history[:, 1], myAgent.state_in: np.vstack(ep_history[:, 0])}
                    grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                    for idx, grad in enumerate(grads):
                        gradBuffer[idx] += grad

                    if current_episode % update_frequency == 0 and current_episode != 0:
                        feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                        _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                        for ix, grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0

                    total_reward.append(running_reward)
                    #total_lenght.append(j)
                    break
                end = timer()
                elapsed+=(end-start)

                    # Update our running tally of scores.
            if current_episode % 100 == 0:
                save_path = saver.save(sess, "save_model/model.ckpt")
                print("Episode: " + str(current_episode) + ", Avg. reward: " + str(np.mean(total_reward[-100:])) + ", Time: " + str(elapsed) + " s")
                elapsed=0

            if not (_ is None):
                with open('scores.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [str(current_episode) + ';' + str(_['rewards'][0]) + ';' + str(_['rewards'][1]) + ';' +str(_['rewards'][2]) + ';' + str(_['rewards'][3])]
                    writer.writerow(row)

            current_episode += 1
    else:
        saver.restore(sess, "save_model/model.ckpt")
        while current_episode < total_episodes:
            s = env.reset()
            s = (s - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            while True:
                a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
                a = np.argmax(a_dist)

                s1, r, d, cause = env.step(a)  # Get our reward for taking an action given a bandit.
                s1 = (s1 - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
                s = s1
                env.render()
                if d == True:
                    total_reward.append(r)
                    break
            if current_episode % 100 == 0:
                print("Episode: " + str(current_episode) + ", Avg. reward: " + str(np.mean(total_reward[-100:])))

            if not (cause is None):
                with open('evaluation.csv', 'a', newline='') as g:
                    writer = csv.writer(g)
                    entry = [str(current_episode) + ';' + str(cause['rewards'][0]) + ';' +
                             str(cause['rewards'][1]) + ';' + str(cause['rewards'][2]) + ';' +
                             str(cause['rewards'][3]) + ';' + str(cause['cause'])]
                    writer.writerow(entry)

            if not (cause is None):
                with open('scores.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [str(current_episode) + ';' + str(cause['rewards'][0]) + ';' + str(cause['rewards'][1]) + ';' +str(cause['rewards'][2]) + ';' + str(cause['rewards'][3])]
                    writer.writerow(row)

            current_episode += 1

