import numpy as np
import gymnasium as gym

#Building the environment
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="ansi")

#Defining the different parameters
epsilon = 0.9
total_episodes = 10000
max_steps = 100
alpha = 0.85
gamma = 0.95
min_exploration_proba = 0.01
exploration_decreasing_decay = 0.001
 
#Initializing the Q-matrix
Q_table = np.zeros((env.observation_space.n, env.action_space.n))

#Function to choose the next action
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_table[state, :])
    return action
 
#Function to learn the Q-value
def update(state, state2, reward, action, action2):
    predict = Q_table[state, action]
    target = reward + gamma * Q_table[state2, action2]
    Q_table[state, action] = Q_table[state, action] + alpha * (target - predict)

#Initializing the reward
reward=0
rewards_per_episode = list()
 
# Starting the SARSA learning
for episode in range(total_episodes):
    t = 0
    state1, info = env.reset()
    action1 = choose_action(state1)

    total_episode_reward = 0
 
    while t < max_steps:
        #Visualizing the training
        # env.render()
         
        #Getting the next state
        state2, reward, terminated, truncated, info = env.step(action1)
 
        #Choosing the next action
        action2 = choose_action(state2)
         
        #Learning the Q-value
        update(state1, state2, reward, action1, action2)
 
        state1 = state2
        action1 = action2
         
        #Updating the respective vaLues
        t += 1
        total_episode_reward = total_episode_reward + reward
         
        #If at the end of learning process
        if terminated or truncated:
            break
    # Need to explore less and less over time (change epsilon)
    epsilon = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*episode))
    rewards_per_episode.append(total_episode_reward)

#Evaluating the performance
print ("Performance : ", reward/total_episodes)
print("Mean reward per thousand episodes")
for i in range(10):
    print((i+1)*1000," mean episode reward: ", np.mean(rewards_per_episode[1000*i:1000*(i+1)]))
 
#Visualizing the Q-matrix
print("\n Final Q-table values:")
print(np.array_str(Q_table, precision=3, suppress_small=True))