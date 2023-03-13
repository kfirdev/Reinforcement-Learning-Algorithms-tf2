#models: policy, target_policy, q_function, target_q_function

#noise_scale=0.1
#polyak=0.995
#action = policy(state) + noise_scale * np.random.randn(n_actions)
#y = r + gamma*(1-done)*target_q_function(new_state, target_policy(new_state))
#q_function_loss = mean( ( q_function(state,action) - y )**2 )
#policy_loss = mean( q_function(state,policy(state)) )
#target_q_function_weight = polyak*target_q_function_weight + (1-polyak)*q_function_weight
#target_policy_weight = polyak*target_policy_weight + (1-polyak)*policy_weight
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
  except RuntimeError as e:
    print(e)


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class ReplayBuffer(object):
    def __init__(self, mem_size, batch_size, obs_dim, n_actions):
        self.mem_size = mem_size
        self.batch_size = batch_size

        self.state_buffer = np.zeros((self.mem_size, obs_dim))
        self.action_buffer = np.zeros((self.mem_size, n_actions))
        self.reward_buffer = np.zeros((self.mem_size, 1))
        self.next_state_buffer = np.zeros((self.mem_size, obs_dim))

        self.obs_dim = obs_dim
        self.ptr, self.size = 0, 0

    def store(self, state, action, reward, new_state, done):

        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.next_state_buffer[self.ptr] = new_state

        self.size = min(self.size+1, self.mem_size)
        self.ptr = (self.ptr+1) % self.mem_size


    def get_data(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[idxs])
        action_batch = tf.convert_to_tensor(self.action_buffer[idxs])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[idxs])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[idxs])

        return state_batch, action_batch, reward_batch, next_state_batch

    def is_train(self):
        if len(self.state_buffer) < self.batch_size:
            return False
        else:
            return True

class create_policy(Model):
	def __init__(self,n_actions):
		super(create_policy, self).__init__()
		last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
		self.d1 = Dense(64, activation='relu')
		self.d2 = Dense(64, activation='relu')
		self.policy_head = Dense(n_actions, activation="tanh", kernel_initializer=last_init)

	def call(self,state):
		hidden = self.d1(state)
		hidden = self.d2(hidden)
		pi = self.policy_head(hidden)
		return pi

class create_q_function(Model):
	def __init__(self):
		super(create_q_function, self).__init__()
		self.d3 = Dense(64, activation='relu')
		self.d4 = Dense(64, activation='relu')
		self.q_head = Dense(1)

	def call(self,state,action):
		action = tf.cast(action, state.dtype)
		qf_h = tf.concat([state, action], axis=1)
		hidden1 = self.d3(qf_h)
		hidden1 = self.d4(hidden1)
		q = self.q_head(hidden1)
		return q


class Agent(object):
	def __init__(self,state_shape,n_actions,std_dev=0.2,lr=0.0003,polyak=0.995,batch_size=64,gamma=0.99,replay_size=1000000,gradient_steps=1,seed=0):
		tf.random.set_seed(seed)
		np.random.seed(seed)

		self.state_shape = state_shape
		self.n_actions = n_actions
		self.std_dev = std_dev
		self.lr = lr
		self.polyak = polyak
		self.gamma = gamma
		self.gradient_steps = gradient_steps

		self.if_start = False
		self.actor_loss = None

		self.policy = create_policy(n_actions)
		self.traget_policy = create_policy(n_actions)

		self.q_function = create_q_function()
		self.target_q_function = create_q_function()

		self.noise_generator = OUActionNoise(mean=np.zeros(n_actions), std_deviation=float(std_dev) * np.ones(1))

		self.buffer = ReplayBuffer(replay_size, batch_size, state_shape, n_actions)

		self.opt = Adam(learning_rate=lr)
		self.q_opt = Adam(learning_rate=lr)

		self.traget_policy.set_weights(self.policy.get_weights())
		self.target_q_function.set_weights(self.q_function.get_weights())

	def predict(self,state):
		state = state[np.newaxis,:]
		
		pi = self.policy(state)

		noise = self.noise_generator()

		action = pi + noise

		action = action[0]

		return action

	def append_data(self, state, new_state, action, reward, done):
		self.buffer.store(state, action, reward, new_state, done)

	def compute_critic_loss(self,state,new_state,action,reward):
		q_value = self.q_function(state,action)

		pi = self.traget_policy(new_state)

		y = reward + self.gamma*self.target_q_function(new_state, pi)

		q_function_loss = tf.math.reduce_mean(tf.math.square(y - q_value))

		return q_function_loss

	def compute_actor_loss(self,state):
		pi = self.policy(state)
		q = self.q_function(state, pi)
		return -tf.reduce_mean(q)

	def soft_update(self,src,trg):
		target_wieghts = []
		for target, source in zip(trg.get_weights(), src.get_weights()):
			fa = self.polyak * target + (1 - self.polyak) * source
			target_wieghts.append(fa)

		trg.set_weights(target_wieghts)



	def train(self, step):
		if not self.buffer.is_train():
			return

		state, action, reward, new_state = self.buffer.get_data()

		for _ in range(self.gradient_steps):

			with tf.GradientTape() as critic_tape:
				critic_loss = self.compute_critic_loss(state,new_state,action,reward)

			critic_grads = critic_tape.gradient(critic_loss, self.q_function.trainable_variables)

			self.q_opt.apply_gradients(zip(critic_grads, self.q_function.trainable_variables))

			with tf.GradientTape() as actor_tape:
				actor_loss = self.compute_actor_loss(state)

			actor_grads = actor_tape.gradient(actor_loss, self.policy.trainable_variables)

			self.opt.apply_gradients(zip(actor_grads, self.policy.trainable_variables))

			self.soft_update(self.policy,self.traget_policy)

			self.soft_update(self.q_function,self.target_q_function)

			self.actor_loss = actor_loss
			self.critic_loss = critic_loss

			target_pred = self.traget_policy(state)

			pi = self.policy(state)
			q_value = self.q_function(state, pi)

			self.q_pred = q_value[0]
			self.pi_pred = pi[0]
			self.target_pred = target_pred[0]







