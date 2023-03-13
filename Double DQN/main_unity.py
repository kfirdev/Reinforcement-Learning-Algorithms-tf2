import matplotlib.pyplot as plt
import numpy as np
import sys
from doubledqn import Agent
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
import pickle
from PIL import Image
from matplotlib import pyplot as plt

#env_name = "D:/pythonprojects/mlenvs/ml2d/ml2d"

#env_name = "D:/pythonprojects/mlenvs/ml2d_camera/ml2d"

env_name = "D:/pythonprojects/mlenvs/ml2d_camera_4agents_working_togheter/ml2d"

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

'''
if len(group_spec.observation_shapes[0]) == 3:
    state_shape = group_spec.observation_shapes[0][0] * group_spec.observation_shapes[0][1] * group_spec.observation_shapes[0][2]
'''

#20_000_000

branch_size = group_spec.discrete_action_branches

def move(action, steps):
    try:
        env.set_actions(group_name, action)
        env.step()
        steps += 1
    except Exception as e:
         print(e)

    step_result = env.get_step_result(group_name)

    new_obs = step_result.obs[0]

    reward = step_result.reward
    done = step_result.done

    return new_obs, reward, done, steps

agent = Agent(state_dims=state_shape, n_actions=branch_size[0], cnn=True)
n_episode = 500_000
steps = 0
score_hist = []
for i in range(5_000):
    done = False
    score = 0
    #env.reset()
    step_result = env.get_step_result(group_name)
    obs = step_result.obs[0]
    step_current = 0
    while not done:
        acts = []

        for s in obs:
            action = agent.get_action(s)
            action = np.array(action)
            acts.append(action)

        acts = np.array(acts).reshape((len(acts),1))

        new_obs, reward, dones, steps = move(acts, steps)

        for a,st,nst,r,d in zip(acts, obs, new_obs,reward, dones):
            agent.store(st,a,r,nst,d)

        d_c = 0
        for dn in dones:
            if dn == True:
                d_c += 1

        if d_c == len(dones):
            done = True


        obs = new_obs
        for re in reward:
            score += re

        if steps % 50 == 0:
            agent.train(i)

    score_hist.append(score)

    avg_score = np.mean(score_hist[-100:])
    print(f'episode {i} score {score} avg score {avg_score} steps {steps} memory_size {agent.memory.get_memory_current_size()}')

    '''if i % 10 == 0 and i != 0:
        agent.save_wieghts()'''
