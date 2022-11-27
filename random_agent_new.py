import gym, cv2
import pandas as pd
import tensorflow as tf

env = gym.make("CartPole-v1", render_mode="rgb_array")

for episode in range(5):
    done = False
    state = env.reset()
    total_rewards = 0
    episode_length = 0
    while not done:
        frame = env.render()
        cv2.imshow("CartPole_Amitabh", frame)
        cv2.waitKey(250)
        # done = True
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32).numpy()
        state, reward, done, truncated, info = env.step(action)

env.close()
