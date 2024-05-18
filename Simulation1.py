import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import os
import re

style.use("ggplot")

SIZE = 20
EPISODES = 10000
MOVE_PENALTY = 1
MOUNTAIN_REWARD = 300  
EPSILON = 0.9
EPSILON_DECAY = 0.9998
SHOW_EVERY = 500
LEARNING_RATE = 0.1
DISCOUNT = 0.95

TARGET = (SIZE - 1, SIZE // 2)

if 'q_table' not in globals():
    q_table = {}
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            q_table[(x1, y1)] = [np.random.uniform(-5, 0) for i in range(4)]

episode_rewards = []
CHARACTER_1 = 1
PASTEL = {1: (0, 255, 0)}

DIRECTORY = "Data S1"
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

class AGENT:
    def __init__(self):
        self.x = 0
        self.y = SIZE - 1

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=0)
        elif choice == 1:
            self.move(x=-1, y=0)
        elif choice == 2:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=0, y=-1)

    def move(self, x=False, y=False):
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

for episode in range(EPISODES):
    character = AGENT()
    
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {EPSILON}")
        show = True
    else:
        show = False
    
    episode_reward = 0
    
    for i in range(200):
        obs = (character.x - TARGET[0], character.y - TARGET[1])
        if np.random.random() > EPSILON:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        character.action(action)

        reward = -MOVE_PENALTY
        if mountain_collision(character.x, character.y, mountain_render):
            reward = MOUNTAIN_REWARD

        new_obs = (character.x - TARGET[0], character.y - TARGET[1])
        max_future_q = np.max(q_table.get(new_obs, np.zeros(4)))
        current_q = q_table[obs][action]
        if reward == MOUNTAIN_REWARD:
            new_q = MOUNTAIN_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q
        
        episode_reward += reward

        if check_if_reached_target(character.x, character.y):
            break
        
        if show:
            env = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)
            mountain_env = env.copy()
            mountain_env[mountain_render > 0] = (0, 0, 255)
            mountain_env = cv2.resize(mountain_env, (300,300), interpolation=cv2.INTER_NEAREST)
            character_position = (int(character.x * 300 / SIZE), int(character.y * 300 / SIZE))
            cv2.circle(mountain_env, character_position, 15, PASTEL[CHARACTER_1], -1)
            cv2.imshow("Mountain Env", mountain_env)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    episode_rewards.append(episode_reward)
    EPSILON *= EPSILON_DECAY

reward_array = np.array(episode_rewards)
smoothed_rewards = np.convolve(reward_array, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot(np.arange(len(smoothed_rewards)), smoothed_rewards)
plt.ylim(0, max(smoothed_rewards) + 20)
plt.xlabel('Episode #')
plt.ylabel('Reward')
plt.savefig(filename)
plt.show()
cv2.destroyAllWindows()