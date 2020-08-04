!pip -q install ./python

#load packages
from unityagents import UnityEnvironment
from maddpg import MADDPGAgent
import numpy as np
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
%matplotlib inline

# instantiate env
env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]

# instantiate agent
agent = MADDPGAgent(state_size=env_info.vector_observations.shape[1], action_size= brain.vector_action_space_size, seed=1)

# Load trained model weights
agent.actor_local.load_state_dict(torch.load('checkpoint_actor1.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic1.pth'))

num_agents = len(env_info.agents)
env_info = env.reset(train_mode=False)[brain_name]      # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
# initialize the score (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = agent.act(states)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

env.close()