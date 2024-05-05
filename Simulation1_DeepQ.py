import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class DQNAgent:
    def __init__(self, input_shape):
        self.state_size = input_shape
        self.action_size = 4  # Four possible actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=self.state_size, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        act_values = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < 32:
            return
        minibatch = random.sample(self.memory, 32)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        self.model.save(filename)

class Environment:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.player = np.array([self.size // 2, self.size // 2])
        return self.get_state()

    def step(self, action):
        if action == 0:  # up
            self.player[0] = max(0, self.player[0] - 1)
        elif action == 1:  # down
            self.player[0] = min(self.size - 1, self.player[0] + 1)
        elif action == 2:  # left
            self.player[1] = max(0, self.player[1] - 1)
        elif action == 3:  # right
            self.player[1] = min(self.size - 1, self.player[1] + 1)

        done = np.array_equal(self.player, [self.size - 1, self.size // 2])
        reward = 100 if done else -1
        return self.get_state(), reward, done

    def get_state(self):
        state = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        state[self.player[0], self.player[1]] = [255, 0, 0]  # Red for player
        state[self.size - 1, self.size // 2] = [0, 255, 0]  # Green for goal
        return state

    def render(self, state):
        img = cv2.resize(state, (300, 300), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Environment', img)
        cv2.waitKey(1)

# Main setup
env = Environment(size=10)
agent = DQNAgent(input_shape=(10, 10, 3))
episodes = 1000
rewards = []

for episode in tqdm(range(1, episodes + 1), ascii=True, unit='episode'):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_memory(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
        env.render(state)

    rewards.append(total_reward)
    if episode % 100 == 0:
        agent.save(f'dqn_episode_{episode}.h5')

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

cv2.destroyAllWindows()