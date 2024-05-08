import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

SIZE = 20
EPISODES = 10000
SHOW_EVERY = 300
MOVE_PENALTY = 1
MOUNTAIN_REWARD = 300
EPSILON_START = 0.9
EPSILON_DECAY = 0.9998
MIN_EPSILON = 0.01
LEARNING_RATE = 0.001
DISCOUNT = 0.95
BATCH_SIZE = 32

DIRECTORY = "EpisodeRewards"
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

def create_environment():
    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    env[SIZE//2:, SIZE//2 - 3:SIZE//2 + 4] = (0, 0, 255)
    return env

def show_environment(env, agent_pos):
    env_copy = env.copy()
    cv2.circle(env_copy, (agent_pos[1], agent_pos[0]), 2, (0, 255, 0), -1)
    img = cv2.resize(env_copy, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Mountain Env", img)
    cv2.waitKey(1)

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=2000)
        self.agent_pos = [SIZE//2, 0]

    def create_model(self):
        model = Sequential([
            Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=(SIZE, SIZE, 3)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(4, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return model

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        current_qs_list = self.model.predict(np.array(states))
        next_qs_list = self.model.predict(np.array(next_states))
        X = []
        y = []
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if not done:
                new_q = reward + DISCOUNT * np.max(next_qs_list[i])
            else:
                new_q = reward
            current_qs = current_qs_list[i]
            current_qs[action] = new_q
            X.append(state)
            y.append(current_qs)
        self.model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE, verbose=0)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def action(self, choice):
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.agent_pos[0] += moves[choice][0]
        self.agent_pos[1] += moves[choice][1]
        self.agent_pos[0] = max(0, min(SIZE - 1, self.agent_pos[0]))
        self.agent_pos[1] = max(0, min(SIZE - 1, self.agent_pos[1]))

env = create_environment()
agent = DQNAgent()
tqdm_bar = tqdm(range(EPISODES), desc='Training Progress')
rewards = []

for episode in tqdm_bar:
    state = env.copy()
    episode_reward = 0
    done = False

    while not done:
        if np.random.random() > agent.epsilon:
            action = np.argmax(agent.get_qs(state))
        else:
            action = np.random.randint(0, 4)
        agent.action(action)
        reward = MOUNTAIN_REWARD if (agent.agent_pos[1] >= SIZE//2 - 3 and agent.agent_pos[1] <= SIZE//2 + 3 and agent.agent_pos[0] >= SIZE//2) else -MOVE_PENALTY
        done = agent.agent_pos[1] >= SIZE//2 - 3 and agent.agent_pos[1] <= SIZE//2 + 3 and agent.agent_pos[0] >= SIZE//2
        new_state = env.copy()
        agent.update_memory(state, action, reward, new_state, done)
        agent.train()
        state = new_state
        episode_reward += reward
        if episode % SHOW_EVERY == 0:
            show_environment(env, agent.agent_pos)

    rewards.append(episode_reward)
    tqdm_bar.set_postfix({'episode_reward': episode_reward})
    agent.epsilon = max(agent.epsilon * EPSILON_DECAY, MIN_EPSILON)

    if episode % SHOW_EVERY == 0 or episode == EPISODES - 1:
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(f"{DIRECTORY}/Episode_rewards_{episode}.jpg")
        plt.close()

cv2.destroyAllWindows()