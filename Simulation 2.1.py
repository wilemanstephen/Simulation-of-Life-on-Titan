# Importing the necessary libraries for our simulation
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")  # Set the plotting style

SIZE = 20  # Size of the grid(world)
EPISODES = 50  # Number of episodes we will run in our simulation
MOVE_PENALTY = 10  # Penalty for moving (to discourage excessive movement)
MOUNTAIN_REWARD = 10  # Reward for reaching the  mountain landmark
LAKE_REWARD = 5 # Reward for reaching the lake landmark
CONSECUTIVE_PENALTY = 7 # Penalty for visiting the lake or the mountain 3 times in a row

EPSILON = 0.99  # Initial exploration rate
EPSILON_DECAY = 0.9995  # Rate at which the exploration rate decays
SHOW_EVERY = 1  # How often to display the simulation (in our case once every 25 episodes)
SKIP = 1

LEARNING_RATE = 0.5  # Learning rate for Q-learning
DISCOUNT = 0.99  # Discount factor for future rewards

# Defining objects in our environment
CHARACTER_1 = 1

# Colors used for rendering
PASTEL = {1: (0, 255, 0)}

# AGENT class defines the behavior of both the character and the landmark
class AGENT:
    def __init__(self):
        while True:
        # Initialize agent (character and landmark) at a random position
            self.x = np.random.randint(0, SIZE // 2)
            self.y = np.random.randint(SIZE // 4, 3 * SIZE // 4)
            
            dist_mountain = np.sqrt((self.x - SIZE//2)**2 + (self.y - SIZE//4)**2)
            dist_lake = np.sqrt((self.x - SIZE//6)**2 + (self.y - SIZE + SIZE//6)**2)

            if np.abs(dist_mountain - dist_lake) < SIZE//10:
                break

            self.consecutive_mountain_visits = 0
            self.consecutive_lake_visits = 0
    
    def reset_consecutive_visits(self):
        self.consecutive_mountain_visits = 0
        self.consecutive_lake_visits = 0
        
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
if 'q_table' not in globals():
    q_table = {}
    # Populate the table with random values for all state-action pairs
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            q_table[(x1, y1)] = [np.random.uniform(-5, 0) for i in range(4)]

episode_rewards = []  # Track rewards per episode

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

def lake(size):
    lake = np.zeros((size, size))
    lake_height = size // 4
    lake_width = size // 4
    lake[-lake_height:, :lake_width] = 1
    return lake

lake_render = lake(SIZE)

def lake_collision(character_x, character_y, lake_grid):
    return lake_grid[int(character_y), int(character_x)] > 0

# Main loop for running episodes
for episode in range(EPISODES):
    character = AGENT()
    mountain_render = mountain(SIZE)
    lake_render = lake(SIZE)

    if episode % SHOW_EVERY == 0:
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):  # Limit the number of steps per episode
        obs = (character - character)  # Observation is the relative position to the landmark
        if np.random.random() > EPSILON:
            # Exploit learned values
            action = np.argmax(q_table[obs])
        else:
            # Explore a new action
            action = np.random.randint(0, 4)
        character.action(action)
        if mountain_collision(character.x, character.y, mountain_render):
            character.consecutive_mountain_visits += 1
            character.consecutive_lake_visits = 0
            if character.consecutive_mountain_visits >= 3:
                reward = CONSECUTIVE_PENALTY
            else:
                reward = MOUNTAIN_REWARD
                print(f"Mountain discovered at episode {episode}, step {i}")
        elif lake_collision(character.x, character.y, lake_render):
            character.consecutive_lake_visits += 1
            character.consecutive_mountain_visits = 0
            if character.consecutive_lake_visits >= 3:
                reward = -CONSECUTIVE_PENALTY
            else:
                reward = LAKE_REWARD
                print(f"Lake discovered at episode {episode}, step {i}")
        else:
            character.reset_consecutive_visits()
            reward = -MOVE_PENALTY

        # Update Q-values using the Q-learning algorithm
        new_obs = (character - character)  # New observation after taking the action
        # Get the maximum Q-value for the new observation
        max_future_q = np.max(q_table.get(new_obs, np.zeros(4))) # Use .get() to avoid KeyError
        current_q = q_table[obs][action]

        # Calculate the new Q-value
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        # Rendering the environment if necessary
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)
            env[mountain_render > 0] = (0, 0, 255)
            env[lake_render > 0] = (255, 0, 0)
            env_resized = cv2.resize(env, (300, 300), interpolation = cv2.INTER_NEAREST)
            character_position = (int(character.x * 300 / SIZE), int(character.y * 300 / SIZE))
            cv2.circle(env_resized, character_position, 15, (0, 255, 0), -1)
            cv2.imshow("Environment", env_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                episode += SKIP
                break

        episode_reward += reward
        if reward == MOUNTAIN_REWARD or reward == LAKE_REWARD:
            break  # Ends the episode if the reward is obtained

    episode_rewards.append(episode_reward)
    EPSILON *= EPSILON_DECAY  # Decay epsilon

    if episode >= EPISODES:
        break

# Plot the rewards over episodes
plt.plot(np.arange(len(episode_rewards)), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()