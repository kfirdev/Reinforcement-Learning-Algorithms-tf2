from dqn import Agent
import gym
import numpy as np

env = gym.make('CartPole-v1')

agent = Agent(n_actions=env.action_space.n,epsilon=1.0,epsilon_dec=0.996,epsilon_min=0.01,gamma=0.99, model=[[256 , 'relu'],[256 , 'relu']], final_layer_activation=None)
# epsilon - the probability of taking random action
# epsilon_dec - the decay of the epsilon (epsilon = epsilon * epsilon_dec)
# epsilon_min - the minimum value of the epsilon
# gamma - the importance of the model prediction on the next state
# model - the model architecture
# final_layer_activation - the activation of the output/last layer

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
        agent.train()

    scores.append(score)
    avg_score = np.mean(scores[max(0, i - 100):(i+1)])
    print(f'episode {i} score {score} avg_score {avg_score} epsilon {agent.epsilon}')
