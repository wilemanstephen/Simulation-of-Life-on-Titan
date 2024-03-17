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
MOVE_PENALTY = 1  # Penalty for moving (to discourage excessive movement)
REWARD = 10  # Reward for reaching the landmark

EPSILON = 0.9  # Initial exploration rate
EPSILON_DECAY = 0.9998  # Rate at which the exploration rate decays
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
        if choice == 0 and self.x < SIZE - 1:
            self.move(x=1, y=0)
        elif choice == 1 and self.x > 0:
            self.move(x=-1, y=0)
        elif choice == 2 and self.y < SIZE - 1:
            self.move(x=0, y=1)
        elif choice == 3 and self.y > 0:
            self.move(x=0, y=-1)
            
    def move(self, x=False, y=False):
        # Move the agent within the bounds of the grid (aka it can't go outside the grid)
        self.x = np.clip(self.x + x, 0, SIZE-1)
        self.y = np.clip(self.y + y, 0, SIZE-1)

# Initialize the Q-table for storing Q-values
if 'q_table_1' not in globals():
    q_table_1 = {}
    # Populate the table with random values for all state-action pairs
    for x in range(SIZE):
        for y in range(SIZE):
            for on_mountain in [True, False]:
                q_table_1[((x,y), on_mountain)] = [np.random.uniform(-2, 0) for _ in range(4)]

if 'q_table_2' not in globals():
    q_table_2 = {}
    for x in range(SIZE):
        for y in range(SIZE):
            for on_mountain in [True, False]:
                q_table_2[((x,y), on_mountain)] = [np.random.uniform(-2, 0) for _ in range(4)]

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

def observation(character, mountain_render):
    char_pos = (character.x, character.y)
    on_mountain = mountain_collision(character.x, character.y, mountain_render)
    state = (char_pos, on_mountain)
    return state

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

    episode_reward1 = 0
    episode_reward2 = 0

    for i in range(200):  # Limit the number of steps per episode
        obs1 = observation(character1, mountain_render)
        obs2 = observation(character2, mountain_render)
        action1 = np.random.randint(0,4) if np.random.random() > EPSILON else np.argmax(q_table_1[obs1])
        action2 = np.random.randint(0,4) if np.random.random() > EPSILON else np.argmax(q_table_2[obs2])
        if not hasattr(character1, 'recent_positions'):
            character1.recent_positions = []
        if not hasattr(character2, 'recent_positions'):
            character2.recent_positions = []

        reward1 = 0
        reward2 = 0
        
        def update_position_memory(agent, new_position):
            memory_size = 5
            agent.recent_positions.append(new_position)
            if len(agent.recent_positions) > memory_size:
                agent.recent_positions.pop(0)
        
        def position_in_memory(agent, position):
            return position in agent.recent_positions
        
        new_pos1 = (character1.x, character1,y)
        new_pos2 = (character2.x, character2.y)

        update_position_memory(character1, new_pos1)
        update_position_memory(character2, new_pos2)

        if position_in_memory(character1, new_pos1):
            reward1 -= MOVE_PENALTY
        
        if position_in_memory(character2, new_pos2):
            reward2 -= MOVE_PENALTY
        
        new_pos1 = (character1.x, character1.y)
        new_pos2 = (character2.x, character2.y)
        character1.action(action1)
        character2.action(action2)
        if mountain_collision(character1.x, character1.y, mountain_render):
            reward1 = REWARD
            print(f"Character 1 achieved reward at episode {episode}, step{i}")
        else:
            reward1 = -MOVE_PENALTY
        if mountain_collision(character2.x, character2.y, mountain_render):
            reward2 = REWARD
            print(f"Character 2 achieved reward at episode {episode}, step{i}")
        else:
            reward2 = -MOVE_PENALTY

        # Update Q-values using the Q-learning algorithm
        max_future_q_1 = np.max(q_table_1[obs1])
        current_q_1 = q_table_1[obs1][action1]
        # Calculate the new Q-value
        new_q_1 = (1 - LEARNING_RATE) * current_q_1 + LEARNING_RATE * (reward1 + DISCOUNT * max_future_q_1)
        
        max_future_q_2 = np.max(q_table_2[obs2])
        current_q_2 = q_table_2[obs2][action2]
        new_q_2 = (1 - LEARNING_RATE) * current_q_2 + LEARNING_RATE * (reward2 + DISCOUNT * max_future_q_2)
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

        episode_reward1 += reward1
        episode_reward2 += reward2
        if reward1 == REWARD or reward2 == episode_rewards:
            break  # Ends the episode if the reward is obtained

    episode_rewards.append(episode_reward1 + episode_reward2)
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