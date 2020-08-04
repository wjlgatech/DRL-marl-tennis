[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Playing Tennis: An Example of Multi-Agent Reinforcement Learning

![Trained Agent][image1]

 
## Objective

To train 2 [MADDPG](https://arxiv.org/abs/1706.02275) agents to play tennis. 

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


## Background

**Environment**: UnityML [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. In this environment, two agents control rackets to bounce a ball over a net.  Thus, the goal of each agent is to keep the ball in play.
**Observation Space**: The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
**Action Space**: Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
**Reward**: If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
To consider the problem to be solved, the agent need to get an average of 30+ over 100 consecutive episodes.

## Getting Start

### Repository

Clone the repository

```
https://github.com/wjlgatech/DRL-marl-tennis.git .
```

### Unity Environment

1. Download the environment that match your system:


- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)


2.  Put the file in the `DRL-continuous-control/` folder, and unzip the file

### Jupyter Notebook

Open the Continuous-control.ipynb notebook

```
jupyter notebook Tennis.ipynb
```

## Code Overview

The code consists of the following modules

```
Continuous_Control.ipynb - the main notebook
ddpg_agent.py - defines the Agent that is to be trained
model.py - defines the ddpg model for the Actor and the Critic network
checkpoint_actor.pth - is the final trained Actor network
checkpoint_critic.pth - is the final trained Critic network
train.py - train the ddpg agent
test.py - test the performance of the trained agent
```

## Results
The average reward collected over 100 episodes is plotted as follow. 

