import pickle
import matplotlib.pyplot as plt

file_path = 'Data S1/Simulation 1_1.pickle'

# Load the pickle file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# If data is a dictionary with episodes as keys and rewards as values, extract them
if isinstance(data, dict):
    episodes = list(data.keys())
    rewards = list(data.values())
    plt.plot(episodes, rewards)
else:
    # If it's a list, we can plot it directly
    plt.plot(data)

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
plt.savefig()