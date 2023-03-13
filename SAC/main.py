import gym
import numpy as np
from sac2 import Agent
import math
import time

if __name__ == '__main__':

    try:
        # Pendulum
        # MountainCarContinuous
        env = gym.make('MountainCarContinuous-v0')
        action_space = env.action_space.shape[0]

        obs_space = env.observation_space.shape[0]
        
        print(action_space, obs_space)
        #2**-13
        agent = Agent(state_shape=obs_space, n_actions=action_space)
        score_hist = []
        step = 0
        n_episode = 500_000
        for i in range(n_episode):
            done = False
            score = 0
            obs = env.reset()
            step_check = 0
            while not done:
                action = agent.select_action(obs)

                new_obs, reward, done, _ = env.step(action)
                
                step += 1
                step_check += 1
                
                '''if i % 50 == 0:
                    env.render()'''
                
                agent.store_data(obs, action, reward, new_obs, done)
                
                obs = new_obs
                score += reward
                
                agent.train(i, step)

            score_hist.append(score)
            #agent.train(i, step)
        
            avg_score = np.mean(score_hist[-100:])
            if agent.if_start:
                print(f'episode {i} score {score} avg score {avg_score}')
                print(f'qf1 loss {agent.qf1_loss} qf2 loss {agent.qf2_loss} policy loss {agent.policy_lossasd} vf loss {agent.valuef_loss} entropy {agent.entropy} step {step_check} total steps {step}')
                print(f'ent_coef {agent.ent_coef} ent_coef_loss {agent.ent_coef_loss} log std {np.mean(agent.log_std)} mu {np.mean(agent.mu)}')
            else:
                print(f'episode {i} score {score} avg score {avg_score}')
            '''if i % 100 == 0:
                agent.save_model()'''

        agent.get_best_hyperparameters()

    except KeyboardInterrupt as e:
        print(e)
        agent.get_best_hyperparameters()
        #agent.key()

