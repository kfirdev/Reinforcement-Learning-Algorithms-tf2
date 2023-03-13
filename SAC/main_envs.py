import numpy as np
from sac2 import Agent
import matplotlib.pyplot as plt
import envs

env = envs.first_env()

obs_space = env.state_shape[0]

action_space = env.n_actions

agent = Agent(state_shape=obs_space, n_actions=action_space)
score_hist = []
n_episode = 200_000
steps = 0
last_score = 0
for i in range(n_episode):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        action = agent.select_action(obs)
        value = agent.value_function(obs.reshape(1, 1))
        print(value)
        new_obs, reward, done = env.env_step(action)
        agent.store_data(obs, action, reward, new_obs, done)
        steps += 1
        obs = new_obs
        score += reward
 
        agent.train(i, steps)

    score_hist.append(score)
    last_score = score

    avg_score = np.mean(score_hist[max(0, i - 100):(i+1)])
    if agent.if_start:
        print(f'episode {i} score {score}')
    else:
        print(f'episode {i} score {round(score, 2)} avg score {avg_score} steps {steps}')
    '''if i % 100 == 0:
        agent.save_model()'''
