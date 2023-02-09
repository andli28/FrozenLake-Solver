# FrozenLake Q-Learner
Implementation of a RL Q-learner using `gymnasium`'s Frozen Lake environment

This was created to play with a RL toy problem, to understand the differences in RL approaches.

## Background
The Q-learning algorithm works well for finite states and actions spaces, but we store every state-action pair in a table, which requires a huge amount of memory.

In the case where states space, actions space or both of them are continuous, it is just impossible to use the Q-learning algorithm.

Note that for Frozen Lakes environments of size 8x8, it becomes more similar to a pathfinding problem, for which RL is actually discouraged and traditional approaches such as A* work better.

## Installation <a name="installation"></a>
**Recommended:** Create a virtual environment and activate it ([`Python Documentation`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/))

```bash
python -m venv env
source env/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

Run:
```bash
python q-solver.py
python sarsa.py
```

## TODO

- Add Online vs. Offline, and other ways for selecting the next action (mean, max, best action)
- Look to add DQN, other RL approaches for different types of games

## Credits
Inspired by: [`TowardsDataScience`](https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187)

Also see: [`GeeksForGeeks: Q-learning in Python`](https://www.geeksforgeeks.org/q-learning-in-python/)
