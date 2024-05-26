import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from collections import deque

np.random.seed(42)
tf.random.set_seed(42)

EPISODES = 1000
SIZE = 10
MOVE_PENALTY = 1
MOUNTAIN_REWARD = 50
EPSILON = 1.0
EPSILON_DECAY = 0.9999
MIN_EPSILON = 0.1
LEARNING_RATE = 0.0005
DISCOUNT = 0.99
MINIBATCH_SIZE = 64
MIN_REPLAY_MEMORY_SIZE = 500

DIRECTORY = "Data S1 DeepQ"
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

def next_run_number(directory):
    files = [f for f in os.listdir(directory) if 'Episode_rewards_' in f]
    return max([int(f.split('_')[-1].split('.')[0]) for f in files] + [0]) + 1

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
            tf.keras.layers.Dense(64, input_shape=(SIZE,), activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(4, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        X, y = [], []
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
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

agent = DQNAgent()
episode_rewards = []
max_reward = float('-inf')
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
        reward = MOVE_PENALTY
        if action == 1 and np.random.rand() < 0.1:
            reward = MOUNTAIN_REWARD
            done = True
        episode_reward += reward
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train()

    max_reward = max(max_reward, episode_reward)
    episode_rewards.append(max_reward)

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

reward_array = np.array(episode_rewards)
plt.figure(facecolor='white')
plt.plot(reward_array, color='red')
plt.xlabel('Episode #')
plt.ylabel('Reward')
plt.savefig(filename)
plt.show()                                                                                                                                                                                                                                                                                                               