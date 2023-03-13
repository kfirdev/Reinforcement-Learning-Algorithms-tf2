import matplotlib.pyplot as plt
import numpy as np
import sys
from sac2 import Agent
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
import pickle

env_name = "D:/pythonprojects/mlenvs/Bouncer/Unity Environment"

#env_name = "D:/pythonprojects/mlenvs/Continuous_Task_real/continuous_task"

work_with_engen = False

seed_count = 10000
seed_pool = [np.random.randint(0, seed_count) for _ in range(seed_count)]

env_seed = np.random.randint(0, 10000)

if work_with_engen:
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(base_port=5004, side_channels=[channel], seed=env_seed, worker_id=0)
    channel.set_configuration_parameters()
else:
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name, side_channels = [engine_configuration_channel], no_graphics=True, seed=env_seed, worker_id=0, docker_training=None)
    engine_configuration_channel.set_configuration_parameters(time_scale=10000, target_frame_rate=100)
# time_scale=10000, target_frame_rate=100
env.reset()


group_name = env.get_agent_groups()[0]
group_spec = env.get_agent_group_spec(group_name)

for dfg in group_spec.observation_shapes[0]:
    state_shape = dfg

try:
    score_final = []
    seed = []
    for s in range(1):
        agent = Agent(state_shape=state_shape, n_actions=group_spec.action_size, seed=env_seed)
        n_episode = 500_000
        score_hist = []
        reward_hist = []
        steps = 0
        i = 0
        while steps <= 20_000_000:
            done = False
            score = 0
            env.reset()
            step_result = env.get_step_result(group_name)
            obs = []
            for o in step_result.obs[0]:
                obs.append(o)

            step_current = 0
            while not done:
                action_list = []
                for asd in obs:
                    action = agent.select_action(asd)
                    action_list.append(action)

                act = np.array(action_list)
                try:
                    env.set_actions(group_name, act)
                    env.step()
                    steps += 1
                    step_current += 1
                except Exception as e:
                    print(e)

                step_result = env.get_step_result(group_name)

                new_obs = []
                for no in step_result.obs[0]:
                new_obs.append(no)

                reward = step_result.reward[0]
                done = step_result.done[0]
                ffor = np.minimum(len(new_obs), len(obs))
                for data in range(ffor):

                    obs_c = obs[data]
                    action_c = action_list[data]
                    reward_c = step_result.reward[data]
                    new_obs_c = new_obs[data]
                    done_c = step_result.done[data]

                    agent.store_data(obs_c, action_c, reward_c, new_obs_c, done_c)


                obs = new_obs
                score += reward
                reward_hist.append(reward)
                try:
                    agent.train(i, steps)
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


            avg_score = np.mean(score_hist)
            avg_reward = np.mean(reward_hist)
            if agent.if_start:
                print(f'episode {i} score {score} avg score {avg_score} avg reward {avg_reward}')
                print(f'qf1 loss {agent.qf1_loss} qf2 loss {agent.qf2_loss} policy loss {agent.policy_lossasd} vf loss {agent.valuef_loss} entropy {agent.entropy} step {step_current} total steps {steps}')
                print(f'ent_coef {agent.ent_coef} ent_coef_loss {agent.ent_coef_loss}')
            else:
                print(f'episode {i} score {score} avg score {avg_score}')

            i += 1

        seed.append(s)

        score_final.append(np.mean(score_hist[-100:]))

except KeyboardInterrupt as e:
    print('ley')
    print(seed[np.argmax(score_final)])
    print(score_final[np.argmax(score_final)])

print(seed[np.argmax(score_final)])
print(score_final[np.argmax(score_final)])
