from doubledqn import Agent
import gym
import numpy as np

env = gym.make('CartPole-v1')

agent = Agent(state_dims=4,n_actions=2)

scores = []
for i in range(5_000):
    done = False
    score = 0
    state = env.reset()
    while not done:
        action = agent.get_action(state)
        new_state, reward, done, _ = env.step(action)
        score += reward
        agent.store(state,action,reward,new_state,done)
        state = new_state
        agent.train(i)

    scores.append(score)
    avg_score = np.mean(scores[max(0, i - 100):(i+1)])
    print(f'episode {i} score %.2f {score} avg_score %.2f {avg_score} epsilon {agent.epsilon}')
