import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
#self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

class ReplayMem(object):
	def __init__(self,mem_size):
		self.mem_size = mem_size
		self.state_memory = []
		self.action_memory = []
		self.reward_memory = []
		self.new_state_memory = []
		self.terminal_state_mem = []
		self.mem_cntr = 0


	def store_data(self,state,action,reward,new_state,done):
		if len(self.state_memory) <= self.mem_size:
			self.state_memory.append(state)
			self.action_memory.append(action)
			self.reward_memory.append(reward)
			self.new_state_memory.append(new_state)
			self.terminal_state_mem.append(done)
			self.mem_cntr += 1

	def sample_buffer(self, batch_size):
		if len(self.new_state_memory) >= batch_size:
			states = self.state_memory[-batch_size:]
			new_states = self.new_state_memory[-batch_size:]
			rewards = self.reward_memory[-batch_size:]
			terminal = self.terminal_state_mem[-batch_size:]
			actions = self.action_memory[-batch_size:]
			return states, new_states, rewards, terminal, actions
		else:
			return 0



class Agent(object):
	def __init__(self,n_actions,epsilon=1.0,epsilon_dec=0.996,epsilon_min=0.01,gamma=0.99, model=[[256 , 'relu'],[256 , 'relu']], final_layer_activation=None):
		self.n_actions = n_actions
		self.epsilon = epsilon
		self.epsilon_dec = epsilon_dec
		self.epsilon_min = epsilon_min
		self.memory = ReplayMem(mem_size=1000000)
		self.q_model = self.make_model()
		self.action_space = [i for i in range(n_actions)]
		self.gamma = gamma
		self.layers = layers
		self.fla = final_layer_activation
		self.batch = 64

	def store(self,state,action,reward,new_state,done):
		#act = np.zeros(self.n_actions)
		#act[action] = 1.0
		self.memory.store_data(state,action,reward,new_state,done)

	def make_model(self):
		model = Sequential()

		for l, a in self.layers:
			model.add(Dense(l,activation=a))

		model.add(Dense(self.n_actions, activation=self.fla))

		model.compile(optimizer=Adam(lr=0.0005),loss='mse')
		return model

	def get_action(self,state):
		state = state[np.newaxis,:]
		if np.random.random() > self.epsilon:
			action_probs = self.q_model.predict(state)
			action = np.argmax(action_probs)
		else:
			action = np.random.choice(self.action_space)

		return action



	def train(self):
		if self.memory.sample_buffer(self.batch) == 0:
			return
		states, new_states, rewards, terminal, actions = self.memory.sample_buffer(self.batch)
		actions = np.array(actions)
		targets_list = []
		states = np.array(states)
		new_states = np.array(new_states)
		q_current = self.q_model.predict(states)
		q_nexts_list = self.q_model.predict(new_states)
		
		for df in range(len(new_states)):
			new_state = np.array([new_states[df]])
			reward = np.array(rewards[df])
			done = np.array(terminal[df])
			state_c = np.array([states[df]])
			act = np.array(actions[df])
			state_preds = np.array(q_current[df])

			q_next = np.array([q_nexts_list[df]])
			target = reward + self.gamma*np.max(q_next, axis=1)*(1 - int(done))

			state_preds[act] = target

			targets_list.append(state_preds)


		targets_list = np.array(targets_list)

		self.q_model.fit(states, targets_list,verbose=0)
		self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min





























































#
