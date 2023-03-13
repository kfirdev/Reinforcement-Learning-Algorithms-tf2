import matplotlib.pyplot as plt
import numpy as np
import sys
from dqn import Agent
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
import pickle

env_name = "D:/pythonprojects/mlenvs/ml2d/ml2d"

#env_name = "D:/pythonprojects/mlenvs/WallJump/Unity Environment"

work_with_engen = False

seed_count = 10000
seed_pool = [np.random.randint(0, seed_count) for _ in range(seed_count)]

env_seed = np.random.randint(0, 10000)

if work_with_engen:
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(base_port=5004, side_channels=[channel], seed=env_seed, worker_id=0)
    channel.set_configuration_parameters(time_scale=20, target_frame_rate=-1)
else:
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name, side_channels = [engine_configuration_channel], no_graphics=False, seed=env_seed, worker_id=0)
    engine_configuration_channel.set_configuration_parameters(time_scale=20, target_frame_rate=-1)
# time_scale=10000, target_frame_rate=100
env.reset()


group_name = env.get_agent_groups()[0]
group_spec = env.get_agent_group_spec(group_name)

for dfg in group_spec.observation_shapes[0]:
    state_shape = dfg

#20_000_000

branch_size = group_spec.discrete_action_branches

try:
    for s in range(1):
        agent = Agent(state_dims=state_shape, n_actions=branch_size[0])
        n_episode = 500_000
        score_hist = []
        reward_hist = []
        steps = 0
        i = 0
        for i in range(5_000):
            done = False
            score = 0
            env.reset()
            step_result = env.get_step_result(group_name)
            obs = step_result.obs[0][0]
            step_current = 0
            while not done:
                #print(obs)
                action = agent.get_action(obs)
                act = np.array([[action]])
                try:
                    env.set_actions(group_name, act)
                    env.step()
                    steps += 1
                    step_current += 1
                except Exception as e:
                    print(e)

                step_result = env.get_step_result(group_name)

                new_obs = step_result.obs[0][0]

                reward = step_result.reward[0]
                done = step_result.done[0]

                agent.store(obs,action,reward,new_obs,done)


                obs = new_obs
                score += reward
                reward_hist.append(reward)
                
            try:
                agent.train()
            except KeyboardInterrupt as e:
                print(seed[np.argmax(score_final)])
                print(score_final[np.argmax(score_final)])
                sys.exit()

            score_hist.append(score)

            '''try:
                agent.train(i, steps)
            except KeyboardInterrupt as e:
                agent.save_model()
                sys.exit()'''

            avg_score = np.mean(score_hist[-100:])
            print(f'episode {i} score {score} avg score {avg_score} steps {steps}')

            i += 1

        seed.append(s)

        score_final.append(np.mean(score_hist[-100:]))

except KeyboardInterrupt as e:
    print('ley')
    print(seed[np.argmax(score_final)])
    print(score_final[np.argmax(score_final)])
