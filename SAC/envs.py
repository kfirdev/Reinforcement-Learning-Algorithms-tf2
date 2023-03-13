import numpy as np

class first_env(object):
	def __init__(self):
		self.time_step = 0
		self.n_actions = 1
		self.state_shape = (1,)
		self.done = False

	def env_step(self,act):
		obs = np.array(0).reshape(self.state_shape)
		reward = 1

		if self.time_step == 0:
			self.done = True

		self.time_step += 1
		return obs, reward, self.done


	def reset(self):
		obs = np.array(0).reshape(self.state_shape)
		self.time_step = 0
		self.done = False
		return obs

class second_env(object):
	def __init__(self):
		self.time_step = 0
		self.n_actions = 1
		self.state_shape = (1,)
		self.done = False

	def env_step(self,act):
		obs = np.array(0).reshape(self.state_shape)
		reward = 0

		if self.time_step == 1:
			self.done = True
			reward = 1

		self.time_step += 1
		return obs, reward, self.done


	def reset(self):
		obs = np.array(0).reshape(self.state_shape)
		self.time_step = 0
		self.done = False
		return obs



