# Importing the necessary libraries for our simulation
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")  # Set the plotting style

SIZE = 10  # Size of the grid(world)
EPISODES = 1000  # Number of episodes we will run in our simulation
MOVE_PENALTY = 1  # Penalty for moving (to discourage excessive movement)
REWARD = 10  # Reward for reaching the landmarkq

EPSILON = 0.99  # Initial exploration rate
EPSILON_DECAY = 0.9995  # Rate at which the exploration rate decays
SHOW_EVERY = 25  # How often to display the simulation (in our case once every 25 episodes)
SKIP = 25

LEARNING_RATE = 0.1  # Learning rate for Q-learning
DISCOUNT = 0.99  # Discount factor for future rewards

# Defining objects in our environment
CHARACTER_1 = 1

# Colors used for rendering
PASTEL = {1: (0, 255, 0)}

# AGENT class defines the behavior of both the character and the landmark
class AGENT:
    def __init__(self):
        # Initialize agent (character and landmark) at a random position
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
        
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
    base_y = np.random.randint(0, size - 3)
    base_x = np.random.randint(0, size - 3)
    mountain[base_x + 1, base_y : base_y + 3] = 1
    mountain [base_x, base_y + 1] = 1
    return mountain

mountain_render = mountain(SIZE)

# Main loop for running episodes
for episode in range(EPISODES):
    character = AGENT()
    mountain_render = mountain(SIZE)

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
        if mountain_render[character.y, character.x] == 1:
            reward = REWARD + np.random.rand()
        else:
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
            env[character.y][character.x] = PASTEL[CHARACTER_1]  # Character location
            mountain_env = env.copy()
            mountain_env[mountain_render > 0] = (0, 0, 255)
            img = Image.fromarray(mountain_env, 'RGB')
            img = img.resize((300, 300), Image.NEAREST)
            cv2.imshow("image", np.array(img))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                episode += SKIP
                break

        episode_reward += reward
        if reward == REWARD:
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
