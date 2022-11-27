'''
Author            : Amitabh Suman
Date              : 26th November, 2022
Action Space      : Discrete(2)
Observation Shape : (4,)
Observation High  : [4.8 inf 0.42 inf]
Observation Low   : [-4.8 -inf -0.42 -inf]
Import            : gym.make("CartPole-v1")
Techniques :
1. SARSA method
2. RL + ANN

'''

import gym, cv2
import numpy as np
import pickle as pkl
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense
# tf.compat.v1.disable_v2_behavior()

env = gym.make("CartPole-v1", render_mode="rgb_array")

done = False
state = env.reset()

# PARAMETERS
# EPSILON = 1
EPSILON = 0.001
EPSILON_DECAY = 1.001
ALPHA = 0.001
GAMMA = 0.99
NUM_EPISODES = 500

# Q Network as this is RL with ANN
net_input = Input(shape=(4,))   # 4 as our observation is a vector of 4 elements
x = Dense(64, activation='relu')(net_input)
x = Dense(32, activation='relu')(x)
output = Dense(2, activation='linear')(x)
q_net = Model(inputs=net_input, outputs=output)


def policy(state, explore=0.0):
    action = tf.argmax(q_net(state)[0], output_type=tf.int32)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)

    return action


for episode in range(NUM_EPISODES):
    done = False
    state = env.reset()
    # print(state)
    # state = float(state[0]) if isinstance(state, tuple) else float(state)
    # print(state)
    # print(np.asarray(state))
    state = tf.convert_to_tensor([np.asarray(state)[0]])
    total_rewards = 0
    episode_length = 0
    while not done:
        action = policy(state, EPSILON)
        next_state, reward, done, truncate, info = env.step(action.numpy())
        next_state = tf.convert_to_tensor([next_state])
        next_action = policy(next_state)

        target = reward + GAMMA * q_net(next_state)[0][next_action]
        if done:
            target = reward

        with tf.GradientTape() as tape:
            current = q_net(state)

        grads = tape.gradient(current, q_net.trainable_weights)
        delta = target - current[0][action]

        for j in range(len(grads)):
            q_net.trainable_weights[j].assign_add(ALPHA * delta * grads[j])

        if episode_length % 2 == 0:
            state = next_state
            action = next_action
        total_rewards += reward
        episode_length += 1

    print("EPISODE : ", episode, "   Episode Length : ", episode_length, "   Total Reward : ", total_rewards, "EPSILON : ", EPSILON)
    # EPSILON /= EPSILON_DECAY

env.close()
# pkl.dump(q_net, "sarsa_q_learning")
q_net.save("q_learning_q_net")