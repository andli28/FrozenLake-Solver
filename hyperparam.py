#number of episodes we will run
n_episodes = 10000

#number of episodes in a round
round_size = 1000

#maximum of iteration per episode
max_iter_episode = 100

#initialize the exploration probability to 1
exploration_proba = 1

#exploration decreasing decay for exponential decreasing
exploration_decreasing_decay = 0.001

#minimum of exploration proba
min_exploration_proba = 0.01

#discounted factor
gamma = 0.99

#learning rate
lr = 0.1