# FrozenLake Q-Learner
Implementation of a RL Q-learner using `gymnasium`'s Frozen Lake environment

This was created to play with a RL toy problem, to understand the differences in RL approaches.

## Background
The Q-learning algorithm works well for finite states and actions spaces, but we store every state-action pair in a table, which requires a huge amount of memory.

In the case where states space, actions space or both of them are continuous, it is just impossible to use the Q-learning algorithm.

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

## TODO

- Make q-solver functional
- Add SARSA (State–action–reward–state–action)
- Add Online vs. Offline, and other ways for selecting the next action (mean, max, best action)

## Credits
Inspired by: [`TowardsDataScience`](https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187)

Also see: [`GeeksForGeeks: Q-learning in Python`](https://www.geeksforgeeks.org/q-learning-in-python/)
