############################# Model Length Hyperparameters #####################################
#number of episodes we will run
n_episodes = 10000

#number of episodes in a round. Therefore, number of rounds is n_episodes/round_size
round_size = 1000

#maximum number of iteration per episode
max_iter_episode = 100

############################# Exploration Hyperparameters #####################################
#initialize the exploration probability to 1. This is equivalent to "1 - the greedy policy", or how often we explore (epsilon)
exploration_proba = 1

#exploration decreasing decay for the exploration exponential decay formula
exploration_decreasing_decay = 0.001

#minimum of exploration probability
min_exploration_proba = 0.01

############################# Model Training Hyperparameters #####################################
#discounted factor (lambda) Future rewards are less valuable than current rewards so they must be discounted.
gamma = 0.99

#learning rate (alpha)
lr = 0.1