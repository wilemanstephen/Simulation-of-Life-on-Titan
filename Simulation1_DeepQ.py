import numpy as np
import cv2
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
import tensorflow as tf
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
    
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=(10, 10, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(4, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
    
    def update_memory(self, state, action, reward, next_state, done):
        pass

    def choose_action(self, state):
        return np.random.randint(0, 4)
    
    def learn(self):
        pass

class Environment:
    def __init__(self, size=10):
        self.SIZE = size
        self.mountain_grid = self.create_mountain(size)
        self.player = np.array([0, size // 2])
        
    def create_mountain(self, size):
        mountain = np.zeros((size, size), dtype=np.uint8)
        peak_height = size // 4
        for i in range(peak_height):
            mountain[size - peak_height + i, (size // 2) - i:(size // 2) + i + 1] = 1
        return mountain
    
    def step(self, action):
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.player += moves[action]
        self.player = np.clip(self.player, 0, self.SIZE - 1)
        reward = 10 if self.mountain_grid[tuple(self.player)] == 1 else -1  # Reward if on mountain
        done = False
        return self.get_image(), reward, done
    
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.mountain_grid == 1] = (0, 255, 0)  # Green for mountain
        env[tuple(self.player)] = (255, 0, 0)  # Red for player
        return env
    
    def render(self):
        img = self.get_image()
        img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        cv2.imshow("Environment", img)
        cv2.waitKey(1)
    
    def reset(self):
        self.player = np.array([0, self.SIZE // 2])
        return self.get_image()

env = Environment()
agent = DQNAgent()
EPISODES = 1000
SHOW_EVERY = 100

episode_rewards = []

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episode'):
    episode_reward = 0
    current_state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(current_state)
        new_state, reward, done = env.step(action)
        agent.update_memory(current_state, action, reward, new_state, done)
        agent.learn()
        current_state = new_state
        episode_reward += reward  # Accumulate reward for the episode
        env.render()

    episode_rewards.append(episode_reward)

    if episode % SHOW_EVERY == 0:
        print(f"Episode: {episode}, Reward: {np.sum(episode_rewards[-SHOW_EVERY:])}")  # Sum for the last SHOW_EVERY episodes

def plot_rewards(ep_rewards):
    plt.plot(np.arange(len(ep_rewards)), ep_rewards)
    plt.xlabel('Episode #')
    plt.ylabel('Reward')
    plt.show()

plot_rewards(episode_rewards)