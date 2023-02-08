import numpy as np
import gymnasium as gym
import hyperparam as hp

# Create the environment: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
env = gym.make("FrozenLake-v1")
n_observations = env.observation_space.n
n_actions = env.action_space.n

# Initialize the Q-table to 0
Q_table = np.zeros((n_observations,n_actions))
#print(Q_table)

rewards_per_episode = list()
exploration_proba = hp.exploration_proba # We initialize the exploration proba to hp.exploration_proba

for e in range(hp.n_episodes):
    # We initialize the first state of the episode
    current_state, info = env.reset()
    if e % 1000 == 0:
        print(Q_table)
    terminated = False
    truncated = False
    
    # Sum the rewards that the agent gets from the environment
    total_episode_reward = 0
    
    for i in range(hp.max_iter_episode): 
        # we sample a float from a uniform distribution over 0 and 1
        # if the sampled flaot is less than the exploration proba
        #     the agent selects a random action
        # else
        #     he exploits his knowledge using the bellman equation 
        
        if np.random.uniform(0,1) < exploration_proba:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[current_state,:])
        
        # The environment runs the chosen action and returns
        # the next state, a reward and true if the episode is ended.
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # We update our Q-table using the Q-learning iteration
        Q_table[current_state, action] = (1-hp.lr) * Q_table[current_state, action] +hp.lr*(reward + hp.gamma*max(Q_table[next_state,:]))
        total_episode_reward = total_episode_reward + reward
        # If the episode is finished, we leave the for loop
        if terminated or truncated:
            break
        current_state = next_state
    #We update the exploration proba using exponential decay formula 
    exploration_proba = max(hp.min_exploration_proba, np.exp(-hp.exploration_decreasing_decay*e))
    rewards_per_episode.append(total_episode_reward)

print("Mean reward per thousand episodes")
for i in range(10):
    print((i+1)*1000," mean episode reward: ", np.mean(rewards_per_episode[1000*i:1000*(i+1)]))

