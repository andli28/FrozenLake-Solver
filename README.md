# FrozenLake Q-Learner
Inspired by: [Link](https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187)

The Q-learning algorithm works well for finite states and actions spaces because - but we store every state-action pair in a table, which requires a huge amount of memory.

In the case where states space, actions space or both of them are continuous, it is just impossible to use the Q-learning algorithm.

## Installation <a name="installation"></a>
**Recommended:** Create a virtual environment and activate it | [`Python Documentation`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

```bash
python -m venv env
source env/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```
