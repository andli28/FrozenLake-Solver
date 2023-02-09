############################# Model Length Hyperparameters #####################################
#number of episodes we will run
n_episodes = 10000

#number of episodes in a round
round_size = 1000

#maximum of iteration per episode
max_iter_episode = 100

############################# Exploration Hyperparameters #####################################
#initialize the exploration probability to 1
exploration_proba = 1

#exploration decreasing decay for the exploration exponential decay formula
exploration_decreasing_decay = 0.001

#minimum of exploration proba
min_exploration_proba = 0.01

############################# Model Training Hyperparameters #####################################
#discounted factor (lambda) Future rewards are less valuable than current rewards so they must be discounted.
gamma = 0.99

#learning rate (alpha)
lr = 0.1