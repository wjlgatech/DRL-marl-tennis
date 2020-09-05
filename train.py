from unityagents import UnityEnvironment
import numpy as np
from maddpg_agent import MADDPGAgent
from collections import deque
from workspaceUtils import keep_awake
import torch


env = UnityEnvironment(file_name="Tennis")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]

agent = MADDPGAgent(state_size=state_size, action_size=action_size, seed=0)

def train(n_episodes=10000, max_t=1000):
    """
    INPUT
    - n_episodes (int): max number of training episodes
    - max_t (int): max time steps for each training episode
    """
    scores = [] # list of average score of all agents for each training episode 
    scores_deque = deque(maxlen=100) # last 100 averaged score for all agents
    scores_mean = [] # list of average score for all agents for 100 consecutive episodes
    
    for i in keep_awake(range(1, n_episodes+1)):
        # reset env
        env_info = env.reset(train_mode=True)[brain_name]
        
        # to store the score for each agent for this episode, ini to 0
        agents_score = np.zeros(num_agents)
        
        # ini state
        states = env_info.vector_observations
        
        # reset the noise process
        agent.reset()
        
        for t in range(max_t):
            # agents take actions according to DDPG policy
            actions = agent.act(states)
            
            # Upon receiving actions, env transits to the next state and provides rewards
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            
            # agent learn from experience tuples
            agent.step(states, actions, rewards, next_states, dones)
            
            # update states
            states = next_states
            
            # accumulative score for each agent
            agents_score += rewards
            
            # check if any of the agent reach the end of episode
            if np.any(dones):
                break
                
        # take the max scores of all agent for this episode
        score_for_this_episode = np.max(agents_score)
        
        scores_deque.append(score_for_this_episode)
        
        final_score = np.mean(scores_deque)
        
        scores.append(score_for_this_episode)
        
        if i % 100 == 0:
            print('\rEpisode {:}\t Average Score {:.2f}'.format(i, final_score, end=''))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor1.pth')
            torch.save(agent.critic_local.state_dict(),'checkpoint_critic1.pth')
        if i >= 100:
            scores_mean.append(final_score)
        if final_score >= 0.5:
            print(f'\nEnvironment solved in {i} episodes. \tAverage Score {final_score}')
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor1.pth')
            torch.save(agent.critic_local.state_dict(),'checkpoint_critic1.pth')
            break
        
        #visualize the performance
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), scores)
        plt.title('MADDPG agent')
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.show()
        
    return scores, scores_mean, agent

scores, scores_mean, agent = train(n_episodes=10000, max_t=1000)

env.close()
