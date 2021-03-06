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

- Command
```
cd Deep_Q-Learning
python main.py --agent=[AGENT_NAME] --env=[ENV_NAME]
```

- Available Agents

    1. DQN: Playing Atari with Deep Reinforcement Learning([논문 링크](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf))

        - First attempt to combine Deep Neural Network with Reinforcement Learning
        - Introduce Experience Replay for sample efficiency and reducing correlation between consecutive samples
        - Fix TD target for stablility


    2. Double DQN: Deep Reinforcement Learning with Double Q-Learning([논문 링크](https://arxiv.org/abs/1509.06461))

        - Alleviate maximization bias via Double Q-Learning
        - Soft update for target update

    3. PER: Prioritized Experience Replay([논문 링크](https://arxiv.org/pdf/1511.05952.pdf))

        - Replay important transitions(large td error) more frequently to learn more efficiently

    4. Dueling DQN: Dueling Network Architecture for Deep Reinforcement Learning([논문 링크](https://arxiv.org/abs/1511.06581))

        - Using Dueling Network to estimate state-value function and action advantage function seperately. It is useful in states where actions do not affect the environment in any relevant way

- Available Environments

    1. Cartpole

        - Discrete Action Space


# Policy Gradient

- Command
```
cd Policy_Gradient
python main.py --agent=[AGENT_NAME] --env=[ENV_NAME]
```

- Available Agents

    1. REINFORCE(Monte-Carlo Policy Gradient)

        - Estimate Policy Gradient by using Monte-Carlo Methods

    2. REIFORCE with Baseline

        - Since REINFORCE algorihm uses Monte-Carlo estimation, it has high variance. To mitigate this issue, we typically subtract a state-dependent baseline(e.x value function) to reduce variance

    3. Advantage Actor-Critic(A2C)

        - Actor-Critic method uses Critic to assist the policy update. A2C algorithm calculates advantage value using critic, which measures whether or not the action is better or worse than policy's default behaviour. It allows us to reduce variance substantially at the cost of introducing bias

    4. Generalized Advantage Estimation(GAE)

        - GAE reduces the varaince of policy gradient estimates at the cost of some bias, with an exponentially-weighted estimator of advantage function

    5. Proximal Policy Optimization(PPO)

        - PPO maximizes a clipped surrogate objective which is simpler to implement than TRPO algorithm

    6. Soft Actor-Critic(SAC)

        - SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework
        
- Available Environments

    1. Cartpole

        - Discrete Action Space  

    2. Pendulum

        - Continuous Action Space
