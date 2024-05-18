import os
import sys
import io
import cv2
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
SHOW_EVERY = 1000  # Changed from 300 to 1000
LEARNING_RATE = 0.001
DISCOUNT = 0.95
TARGET = (SIZE - 1, SIZE // 2)
CHARACTER_1 = 1
PASTEL = {1: (0, 255, 0)}

DIRECTORY = "DATA_S1_DeepQ"
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

class AGENT:
    def __init__(self):
        self.x = 0
        self.y = SIZE - 1

    def action(self, choice):
        if choice == 0:
            self.move(1, 0)
        elif choice == 1:
            self.move(-1, 0)
        elif choice == 2:
            self.move(0, 1)
        elif choice == 3:
            self.move(0, -1)

    def move(self, x, y):
        self.x = np.clip(self.x + x, 0, SIZE - 1)
        self.y = np.clip(self.y + y, 0, SIZE - 1)

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

def mountain_collision(character_x, character_y, mountain_grid):
    return mountain_grid[character_y, character_x] > 0

def check_if_reached_target(character_x, character_y):
    return mountain_collision(character_x, character_y, mountain_render)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(4, activation='linear')
])
model.compile(optimizer='adam', loss='mse')

target_model = tf.keras.models.clone_model(model)
target_model.set_weights(model.get_weights())
replay_memory = deque(maxlen=20000)

episode_rewards = []
for episode in tqdm(range(EPISODES), ascii=True, unit='episodes'):
    character = AGENT()
    episode_reward = 0
    for i in range(200):
        obs = np.array([character.x - TARGET[0], character.y - TARGET[1]])
        if np.random.random() > EPSILON:
            action = np.argmax(model.predict(obs.reshape(1, -1))[0])
        else:
            action = np.random.randint(0, 4)
        character.action(action)
        reward = MOUNTAIN_REWARD if mountain_collision(character.x, character.y, mountain_render) else -MOVE_PENALTY
        done = check_if_reached_target(character.x, character.y)
        episode_reward += reward
        if done:
            break
        if episode % SHOW_EVERY == 0:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[mountain_render == 1] = (0, 0, 255)
            env[character.y, character.x] = PASTEL[CHARACTER_1]
            env = cv2.resize(env, (300, 300), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Game', env)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    EPSILON *= EPSILON_DECAY
    episode_rewards.append(episode_reward)

cv2.destroyAllWindows()

reward_array = np.array(episode_rewards)
smoothed_rewards = np.convolve(reward_array, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
plt.plot(np.arange(len(smoothed_rewards)), smoothed_rewards)
plt.ylim(0, max(smoothed_rewards) + 20)
plt.xlabel('Episode #')
plt.ylabel('Reward')
plt.savefig(filename)
plt.show()