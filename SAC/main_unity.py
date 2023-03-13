import matplotlib.pyplot as plt
import numpy as np
import sys
from sac2 import Agent
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
import pickle

env_name = "D:/pythonprojects/mlenvs/BouncerSingle/Unity Environment"

#env_name = "D:/pythonprojects/mlenvs/contius_task/continuous_task"

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

def move(action, steps):
    try:
        env.set_actions(group_name, action)
        env.step()
        steps += 1
    except Exception as e:
         print(e)

    step_result = env.get_step_result(group_name)

    new_obs = step_result.obs[0][0]

    reward = step_result.reward[0]
    done = step_result.done[0]

    return new_obs, reward, done, steps

agent = Agent(state_shape=state_shape, n_actions=group_spec.action_size, seed=env_seed,scale_reward=5)
n_episode = 500_000
score_hist = []
reward_hist = []
steps = 0
for i in range(n_episode):
    done = False
    score = 0
    env.reset()
    step_result = env.get_step_result(group_name)
    obs = step_result.obs[0][0]
    step_current = 0
    while not done:
        action = agent.select_action(obs)
        act = np.array([action])

        new_obs, reward, done, steps = move(act, steps)

        agent.store_data(obs, action, reward, new_obs, done)

        obs = new_obs
        score += reward

        agent.train(i, steps)

    score_hist.append(score)
    avg_score = np.mean(score_hist[-100:])

    if agent.if_start:
        print(f'episode {i} score {score} avg score {avg_score}')
        print(f'qf1 loss {agent.qf1_loss} qf2 loss {agent.qf2_loss} policy loss {agent.policy_lossasd} vf loss {agent.valuef_loss} entropy {agent.entropy} step {step_current} total steps {steps}')
        #print(f'ent_coef {agent.ent_coef} ent_coef_loss {agent.ent_coef_loss} mu {agent.mu}')
        print(f'ent_coef {agent.ent_coef} ent_coef_loss {agent.ent_coef_loss} log std {np.mean(agent.log_std)} mu {np.mean(agent.mu)}')
    else:
        print(f'episode {i} score {score} avg score {avg_score}')

