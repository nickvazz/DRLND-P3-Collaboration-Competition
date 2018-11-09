import os
import argparse
import json

import torch
from ddpg_agent import Agents
from collections import deque
from unityagents import UnityEnvironment

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()
parser.add_argument('--n_episodes', default=400)
parser.add_argument('--max_t', default=1000)
parser.add_argument('--BUFFER_SIZE', default=1e5, type=int)
parser.add_argument('--BATCH_SIZE', default=64, type=int)
parser.add_argument('--GAMMA', default=0.99)
parser.add_argument('--TAU', default=1e-3)
parser.add_argument('--LR_ACTOR', default=1e-4)
parser.add_argument('--LR_CRITIC', default=1e-4)
parser.add_argument('--CRITIC_WEIGHT_DECAY', default=0)
parser.add_argument('--fc1_units', default=400)
parser.add_argument('--fc2_units', default=300)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--model_num', default=1)



args = vars(parser.parse_args())
print(args)

for key, value in args.items():
    exec(f'{key} = {value}')

os.system(f'mkdir -p results/model-{model_num}')
with open(f'results/model-{model_num}/training_params.json', 'w') as outfile:
    json.dump(args, outfile)

# env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64", worker_id=int(f'4{model_num}'))
# env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64", worker_id=int(f'{model_num}'))
# env = UnityEnvironment(file_name="Soccer_Linux/Soccer.x86_64", worker_id=int(f'5{model_num}'))
# env = UnityEnvironment(file_name="Soccer_Linux_NoVis/Soccer.x86_64", worker_id=int(f'5{model_num}'))
env = UnityEnvironment(file_name="Tennis.app", worker_id=1000)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

agent = Agents(state_size=state_size, action_size=action_size, num_agents=num_agents,
               random_seed=seed, fc1_units=fc1_units, fc2_units=fc2_units, BUFFER_SIZE=BUFFER_SIZE,
               BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, TAU=TAU, LR_ACTOR=LR_ACTOR,
               LR_CRITIC=LR_CRITIC, CRITIC_WEIGHT_DECAY=CRITIC_WEIGHT_DECAY)

def ddpg(n_episodes=400, max_t=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(state, action, rewards, next_state, dones)
            state = next_state
            score += rewards
            if np.any(dones):
                break

        # print (score)
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.3f}'.format(i_episode, np.mean(scores_deque), np.mean(score)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), f'results/model-{model_num}/actor_{i_episode}.pth')
            torch.save(agent.critic_local.state_dict(), f'results/model-{model_num}/critic_{i_episode}.pth')
    return scores


scores = ddpg(n_episodes, max_t)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('results/model-{}/scores.png'.format(model_num))
# plt.show()

df = pd.DataFrame({'episode':np.arange(len(scores)),'score':scores})
df.set_index('episode', inplace=True)
df.to_csv('results/model-{}/scores.csv'.format(model_num))

os.system('cp model.py results/model-{}/'.format(model_num))

env.close()
