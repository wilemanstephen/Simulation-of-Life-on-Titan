# Importing the necessary libraries for our simulation
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import os
import re

style.use("ggplot")  # Set the plotting style

SIZE = 20  # Size of the grid(world)
EPISODES = 50  # Number of episodes we will run in our simulation
MOVE_PENALTY = 4  # Penalty for moving (to discourage excessive movement)
REWARD = 10  # Reward for reaching the landmark

EPSILON = 0.99  # Initial exploration rate
EPSILON_DECAY = 0.9995  # Rate at which the exploration rate decays
SHOW_EVERY = 1  # How often to display the simulation (in our case once every 25 episodes)
SKIP = 1

LEARNING_RATE = 0.5  # Learning rate for Q-learning
DISCOUNT = 0.99  # Discount factor for future rewards

# Defining objects in our environment
CHARACTER_1 = 1
CHARACTER_2 = 2

# Colors used for rendering
PASTEL = {1: (0, 255, 0), 2: (255, 0, 0)}

# AGENT class defines the behavior of both the character and the landmark
class AGENT:
    def __init__(self):
        # Initialize agent (character and landmark) at a random position
        self.x = np.random.randint(0, SIZE // 2)
        self.y = np.random.randint(SIZE // 4, 3 * SIZE // 4)
        
    def __str__(self):
        # String representation for debugging
        return f"{self.x}, {self.y}"
        
    def __sub__(self, other):
        # Gets the relative position of another agent
        return (self.x - other.x, self.y - other.y)
        
    def action(self, choice):
        # The character executes movement based on action choice
        if choice == 0:
            self.move(x=1, y=0)
        elif choice == 1:
            self.move(x=-1, y=0)
        elif choice == 2:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=0, y=-1)
            
    def move(self, x=False, y=False):
        # Move the agent within the bounds of the grid (aka it can't go outside the grid)
        self.x = np.clip(self.x + x, 0, SIZE-1)
        self.y = np.clip(self.y + y, 0, SIZE-1)

# Initialize the Q-table for storing Q-values
if 'q_table_1' not in globals():
    q_table_1 = {}
    # Populate the table with random values for all state-action pairs
    for x in range(-SIZE + 1, SIZE):
        for y in range(-SIZE + 1, SIZE):
            q_table_1[(x, y)] = [np.random.uniform(-5, 0) for i in range(4)]

if 'q_table_2' not in globals():
    q_table_2 = {}
    for x in range(-SIZE + 1, SIZE):
        for y in range(-SIZE + 1, SIZE):
            q_table_2[(x,y)] = [np.random.uniform(-5, 0) for i in range(4)]

episode_rewards = []  # Track rewards per episode

DIRECTORY = "Data S2.2" # Directory where all Q-Tables to be saved
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

#This function has the role of adding a number at the end of each iteration of the simulation for ease of understanding.
def next_run_number(directory, character_name):
    max_num = 0
    for filename in os.listdir(directory):
        match = re.match(rf"{character_name}_Simulation 2.2_(\d+).pickle", filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num + 1

def mountain(size):
    mountain = np.zeros((size, size))

    # Parameters for mountain size
    peak_height = size // 4  # Make the peak height smaller
    base_width = size // 4  # Make the base width smaller

    # Starting point for the base of the mountain at the top right
    base_start_col = size - base_width

    # Create the pyramid from top to bottom
    for i in range(peak_height):
        # Calculate the starting and ending point for each row
        start = base_start_col + (peak_height - i - 1)
        end = size - (peak_height - i - 1)
        mountain[i, start:end] = 1

    return mountain

mountain_render = mountain(SIZE)

def mountain_collision(character_x, character_y, mountain_grid):
    return mountain_grid[int(character_y), int(character_x)] > 0

def distance_mountain(character, mountain_render):
    distance = [np.sqrt((character.x - col)**2 + (character.y - row)**2) for row in range(SIZE) for col in range(SIZE) if mountain_render[row, col] == 1]
    return min(distance) if distance else SIZE

def observation(character, mountain_render):
    for row in range(SIZE):
        for col in range(SIZE):
            if mountain_render[row, col] == 1:
                return (character.x - col, character.y - row)
    return(0, 0)

character1_run_number = next_run_number(DIRECTORY, 'character1')
character2_run_number = next_run_number(DIRECTORY, 'character2')

# Main loop for running episodes
for episode in range(EPISODES):
    character1 = AGENT()
    while True:
        character2 = AGENT()
        if(character1.x != character2.x) or (character1.y != character2.y):
            break
    mountain_render = mountain(SIZE)

    if episode % SHOW_EVERY == 0:
        show = True
    else:
        show = False

    episode_reward = 0

    for i in range(200):  # Limit the number of steps per episode
        obs1 = observation(character1, mountain_render)
        obs2 = observation(character2, mountain_render)
        action1 = np.random.randint(0,4) if np.random.random() > EPSILON else np.argmax(q_table_1[obs1])
        action2 = np.random.randint(0,4) if np.random.random() > EPSILON else np.argmax(q_table_2[obs2])
        character1.action(action1)
        character2.action(action2)
        if mountain_collision(character1.x, character1.y, mountain_render):
            reward = REWARD
            print(f"Character 1 achieved reward at episode {episode}, step{i}")
        else:
            reward = -MOVE_PENALTY
        if mountain_collision(character2.x, character2.y, mountain_render):
            reward = REWARD
            print(f"Character 2 achieved reward at episode {episode}, step{i}")
        else:
            reward = -MOVE_PENALTY

        # Update Q-values using the Q-learning algorithm
        max_future_q_1 = np.max(q_table_1[obs1])
        current_q_1 = q_table_1[obs1][action1]
        # Calculate the new Q-value
        new_q_1 = (1 - LEARNING_RATE) * current_q_1 + LEARNING_RATE * (reward + DISCOUNT * max_future_q_1)
        
        max_future_q_2 = np.max(q_table_2[obs2])
        current_q_2 = q_table_2[obs2][action2]
        new_q_2 = (1 - LEARNING_RATE) * current_q_2 + LEARNING_RATE * (reward + DISCOUNT * max_future_q_2)

        reward_1 = REWARD - distance_mountain(character1, mountain_render)
        reward_2 = REWARD - distance_mountain(character2, mountain_render)

        # Rendering the environment if necessary
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            mountain_env = env.copy()
            mountain_env[mountain_render > 0] = (0, 0, 255)
            mountain_env = cv2.resize(mountain_env, (300,300), interpolation = cv2.INTER_NEAREST)
            env[character1.y][character1.x] = PASTEL[CHARACTER_1]
            env[character2.y][character2.x] = PASTEL[CHARACTER_2]
            character1_position = (int(character1.x * 300 / SIZE), int(character1.y * 300 / SIZE))
            character2_position = (int(character2.x * 300 / SIZE), int(character2.y * 300 / SIZE))
            cv2.circle(mountain_env, character1_position, 15, PASTEL[CHARACTER_1], -1)
            cv2.circle(mountain_env, character2_position, 15, PASTEL[CHARACTER_2], -1)
            cv2.imshow("image", mountain_env)
            if cv2.waitKey(200) & 0xFF == ord('q'):
                episode += SKIP
                break

        episode_reward += reward
        if reward == REWARD:
            break  # Ends the episode if the reward is obtained

    episode_rewards.append(episode_reward)
    EPSILON *= EPSILON_DECAY  # Decay epsilon
    if EPSILON < 0.1:
        EPSILON = 0.1

    if episode >= EPISODES - 1:
        with open(os.path.join(DIRECTORY, f"character1_Simulation 2.2_{character1_run_number}.pickle"), 'wb') as f:
            pickle.dump(q_table_1, f) #saves the table inside the directory
        with open(os.path.join(DIRECTORY, f"character2_S2.2_{character2_run_number}.pickle"), 'wb') as f:
            pickle.dump(q_table_2, f)

# Plot the rewards over episodes
plt.plot(np.arange(len(episode_rewards)), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()