import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import os
import re

style.use("ggplot")

SIZE = 10
EPISODES = 10000
MOVE_PENALTY = 1
REWARD = 10

EPSILON = 0.9
EPSILON_DECAY = 0.1
SHOW_EVERY = 25

LEARNING_RATE = 0.1
DISCOUNT = 0.95

CHARACTER_1 = 1
LANDMARK_1 = 2

PASTEL = {1: (0, 255, 0), 2: (224, 255, 255)}

DIRECTORY = "Data S1 (Rough)"
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
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
        
    def __str__(self):
        return f"{self.x}, {self.y}"
        
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
        
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
        self.x = np.clip(self.x + x, 0, SIZE-1)
        self.y = np.clip(self.y + y, 0, SIZE-1)

if 'q_table' not in globals():
    q_table = {}
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            q_table[(x1, y1)] = [np.random.uniform(-5, 0) for i in range(4)]

episode_rewards = []

for episode in range(EPISODES):
    character = AGENT()
    landmark = AGENT()

    if episode % SHOW_EVERY == 0:
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200): 
        obs = (character - landmark) 
        if np.random.random() > EPSILON:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        character.action(action)

        if character.x == landmark.x and character.y == landmark.y:
            reward = REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (character - landmark)
        max_future_q = np.max(q_table.get(new_obs, np.zeros(4)))
        current_q = q_table[obs][action]

        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[landmark.y][landmark.x] = PASTEL[LANDMARK_1]
            env[character.y][character.x] = PASTEL[CHARACTER_1]
            img = Image.fromarray(env, 'RGB')
            img = img.resize((300, 300), Image.NEAREST)
            cv2.imshow("Simulation", np.array(img))
            if reward == REWARD or cv2.waitKey(5) & 0xFF == ord('q'):
                break

        episode_reward += reward
        if reward == REWARD:
            break

    episode_rewards.append(episode_reward)
    EPSILON *= EPSILON_DECAY

plt.plot(np.arange(len(episode_rewards)), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig(filename)
plt.show()
cv2.destroyAllWindows()