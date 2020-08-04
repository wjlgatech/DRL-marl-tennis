[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Playing Tennis: An Example of Multi-Agent Reinforcement Learning

![Trained Agent][image1]

 
## Objective

To train 2 agents to play tennis. 

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


## Background

**Environment**: [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, which is a modified version of Unity ML [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis). In this environment, two agents control rackets to bounce a ball over a net.  Thus, the goal of each agent is to keep the ball in play.

**Observation Space**: The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. 

**Action Space**: Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

**Reward**: If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
To consider the problem to be solved, the agent need to get an average of 30+ over 100 consecutive episodes.

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 


### Repository

Clone the repository

```
https://github.com/wjlgatech/DRL-marl-tennis.git .
```

### How to run the codes

One way is that you open the Tennis.ipynb notebook to follow instructions there:

```
jupyter notebook Tennis.ipynb
```


Another way is that you train the agents with

```
python train.phy
```

and test the trained agents with
```
python test.py
```

## Code Overview

The code consists of the following modules

```
Tennis.ipynb - the main notebook
MADDPG.py - defines the Agent that is to be trained
model.py - defines the MADDPG model for the Actor and the Critic network
checkpoint_actor1.pth - is the final trained Actor network
checkpoint_criti1c.pth - is the final trained Critic network
train.py - train the MADDPG agent
test.py - test the performance of the trained agent
```

## Results
Environment solved in 6800 episodes with	Average Score 0.515900007802993 (>=0.5).
