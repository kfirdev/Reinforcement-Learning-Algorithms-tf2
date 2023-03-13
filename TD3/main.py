import gym
import numpy as np
from TD3 import Agent
import matplotlib.pyplot as plt

#MountainCar
#CartPole
#Pendulum
#MountainCarContinuous
env = gym.make('Pendulum-v0')
action_space = env.action_space.shape[0]

obs_space = env.observation_space.shape[0]

print(action_space, obs_space)
#2**-13
agent = Agent(state_shape=obs_space, n_actions=action_space)
score_hist = []
log_std_hist_y = []
step_hist_x = []
n_episode = 200_000
steps = 0
last_score = 0
for i in range(n_episode):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        action = agent.predict(obs)
        new_obs, reward, done, _ = env.step(action)
        agent.append_data(obs, action, reward, new_obs)
        steps += 1
        obs = new_obs
        score += reward
 
        agent.train(steps)
        
    score_hist.append(score)
    last_score = score

    avg_score = np.mean(score_hist[max(0, i - 100):(i+1)])
    if agent.policy_loss is not None:
        print(f'episode {i} score {score} avg score {avg_score} steps {steps} actor loss {agent.policy_loss.numpy()} critic loss {agent.critic_loss.numpy()}')
    else:
        print(f'episode {i} score {round(score, 2)} avg score {avg_score} steps {steps}')
    '''if i % 100 == 0:
        agent.save_model()'''
