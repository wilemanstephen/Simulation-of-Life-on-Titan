import os
import sys
import io
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from tqdm import tqdm
import tensorflow as tf
from collections import deque
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
style.use("ggplot")

SIZE = 20
EPISODES = 10000
MOVE_PENALTY = 1
MOUNTAIN_REWARD = 300
EPSILON = 0.9
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.01
LEARNING_RATE = 0.001
DISCOUNT = 0.95
MINIBATCH_SIZE = 64
MIN_REPLAY_MEMORY_SIZE = 1000
SHOW_EVERY = 100

DIRECTORY = "Data S1 DeepQ"
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

def next_run_number(directory):
    max_num = 0
    for filename in os.listdir(directory):
        match = re.match(r"Episode_rewards_(\d+).jpg", filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num + 1

run_number = next_run_number(DIRECTORY)
filename = f"{DIRECTORY}/Episode_rewards_{run_number}.jpg"

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=20000)

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_shape=(SIZE,), activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(4, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        X = []
        y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            y.append(current_qs)
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False if terminal_state else None)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

def mountain(size):
    mountain = np.zeros((size, size))
    peak_height = size // 4
    base_width = size // 4
    base_start_col = size - base_width
    for i in range(peak_height):
        start = base_start_col + (peak_height - i - 1)
        end = size - (peak_height - i - 1)
        mountain[i, start:end] = 1
    return mountain

mountain_render = mountain(SIZE)

def check_if_reached_target(character_x, character_y):
    return mountain_render[character_y, character_x] == 1

agent = DQNAgent()
episode_rewards = []
for episode in tqdm(range(EPISODES), ascii=True, unit='episodes'):
    episode_reward = 0
    current_state = np.random.randint(0, SIZE, (SIZE,))
    done = False

    while not done:
        if np.random.random() > EPSILON:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, 4)
        new_state = current_state.copy()
        if action == 1:
            reward = MOUNTAIN_REWARD if check_if_reached_target(current_state[0], current_state[1]) else -MOVE_PENALTY
            done = True
        else:
            reward = -MOVE_PENALTY
        episode_reward += reward
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)
        current_state = new_state

    episode_rewards.append(episode_reward)
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
    else:
        EPSILON = MIN_EPSILON

reward_array = np.array(episode_rewards)
smoothed_rewards = np.convolve(reward_array, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
plt.figure(facecolor='white')
plt.plot(np.arange(len(smoothed_rewards)), smoothed_rewards, color='red')
plt.ylim(min(smoothed_rewards) - 20, max(smoothed_rewards) + 20)
plt.xlabel('Episode #')
plt.ylabel('Reward')
plt.savefig(filename)
plt.show()