# RL_implementations
Implementation of RL Algorithms(pytorch)

# Installation
- Clone my github repo
```
git clone https://github.com/dbsxodud-11/RL_implementations.git
cd RL_implementations
```
- Install conda and create new environment using `environment.yml` file
```
conda create -f environment.yml
```

# Deep Q-Learning

1. Playing Atari with Deep Reinforcement Learning([논문 링크](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf))

    - First attempt to combine Deep Neural Network with Reinforcement Learning
    - Introduce Experience Replay for sample efficiency and reducing correlation between consecutive samples
    - Fix TD target for stablility


2. Deep Reinforcement Learning with Double Q-Learning([논문 링크](https://arxiv.org/abs/1509.06461))

    - Alleviate maximization bias via Double Q-Learning
    - Soft update for target update

3. Prioritized Experience Replay([논문 링크](https://arxiv.org/pdf/1511.05952.pdf))

    - Replay important transitions(large td error) more frequently to learn more efficiently

4. Dueling Network Architecture for Deep Reinforcement Learning([논문 링크](https://arxiv.org/abs/1511.06581))

    - Using Dueling Network to estimate state-value function and action advantage function seperately. It is useful in states where actions do not affect the environment in any relevant way
