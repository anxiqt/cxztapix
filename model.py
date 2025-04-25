#DQN model instead > Greedy Model
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Define Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Hyperparameters
ENV_NAME = "CartPole-v1"
EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
LR = 0.001
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

# Experience Replay Memory
memory = deque(maxlen=MEMORY_SIZE)

# Environment setup
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize networks
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return policy_net(state_tensor).argmax().item()

def replay():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    current_q = policy_net(states).gather(1, actions)
    next_q = target_net(next_states).max(1)[0].detach().unsqueeze(1)
    expected_q = rewards + (GAMMA * next_q * (1 - dones))

    loss = loss_fn(current_q, expected_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    done = False
    while not done:
        action = select_action(state, EPSILON)
        next_state, reward, done, truncated, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        replay()

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {EPSILON:.3f}")

env.close()