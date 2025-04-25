# here we will define the reinforcment model we will use to train the model

# q-learning model epsilon-greedy strategy

import gym
import numpy as np
import random
from collections import defaultdict

#create gym environment (i.e. FrozenLake)

env = gym.make("FrozenLake-v1", is_slippery=True)


#initialize Q table with zeros
q_table = defaultdict(lambda: np.zeros(env.action_space.n))

#parameters: learning rate, discount factor (greedy), exploration, decay, episdoes, steps...
alpha = 0.1
gamme = 0.99
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01
episodes = 5000
max_steps = 100

# training 
for episode in range(episodes):
    state,_ = env.reset()
    done = False

    for _ in range(max_steps):
        #exploration
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample() #xplore
        else:
            action = np.argmax(q_table[state]) #xploit

        next_state, reward, done, truncated, _  = env.step(action)

        #update q-value...
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state][action] = new_value

        state = next_state

        if done:
            break

    

    #decay
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training completed. \n")


# test
state, _ = env.reset()
env.render()


done = False
while not done:
    action = np.argmax(q_table[state])
    state, reward, done, truncated, _ = env.step(action)
    env.render()

print(f"Reward: {reward}")
