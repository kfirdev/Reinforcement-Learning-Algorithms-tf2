#models: policy, target_policy, 2 q_fuctions, 2 target q_fuctions
#target_noise = 0.2
#noise_clip = 0.5
#pi_targ = target_policy(new_state)
#epsilon = tf.random.normal(tf.shape(pi_targ), stddev=target_noise)
#epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
#target_action = pi_targ + epsilon
'''
def get_action(state):
    a = policy(state) + np.random.randn(n_actions)
    return a
'''
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
		tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
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

    def store(self, state, action, reward, new_state):

        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.next_state_buffer[self.ptr] = new_state

        self.size = min(self.size+1, self.mem_size)
        self.ptr = (self.ptr+1) % self.mem_size


    def get_data(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)

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
		self.d1 = Dense(64, activation='relu')
		self.d2 = Dense(64, activation='relu')
		self.policy_head = Dense(n_actions, activation='tanh')

	def call(self,state):
		o1 = self.d1(state)
		o1 = self.d2(o1)
		pi = self.policy_head(o1)
		return pi

class create_q_function(Model):
	def __init__(self):
		super(create_q_function, self).__init__()
		self.d1 = Dense(64, activation='relu')
		self.d2 = Dense(64, activation='relu')
		self.q_head = Dense(1)

		self.d3 = Dense(64, activation='relu')
		self.d4 = Dense(64, activation='relu')
		self.sec_q_head = Dense(1)

	def call(self,state,action):
		action = tf.cast(action, state.dtype)
		qf_h = tf.concat([state, action], axis=1)

		o1 = self.d1(qf_h)
		o1 = self.d2(o1)
		q1 = self.q_head(o1)

		o2 = self.d3(qf_h)
		o2 = self.d4(o2)
		q2 = self.sec_q_head(o2)

		return q1, q2


class Agent(object):
	def __init__(self,state_shape,n_actions,policy_train_every=2,noise_clip=0.5,std_dev=0.2,lr=0.0003,polyak=0.995,batch_size=64,gamma=0.99,replay_size=1000000,gradient_steps=1,seed=0):
		tf.random.set_seed(seed)
		np.random.seed(seed)

		self.state_shape = state_shape
		self.n_actions = n_actions
		self.std_dev = std_dev
		self.lr = lr
		self.polyak = polyak
		self.gamma = gamma
		self.gradient_steps = gradient_steps
		self.policy_train_every = policy_train_every
		self.noise_clip = noise_clip

		self.policy_loss = None

		self.policy = create_policy(n_actions)
		self.target_policy = create_policy(n_actions)

		self.q_function = create_q_function()
		self.target_q_function = create_q_function()

		self.noise_generator = OUActionNoise(mean=np.zeros(n_actions), std_deviation=float(std_dev) * np.ones(1))

		self.buffer = ReplayBuffer(replay_size, batch_size, state_shape, n_actions)

		self.opt = Adam(learning_rate=lr)
		self.q_opt = Adam(learning_rate=lr)

		self.target_policy.set_weights(self.policy.get_weights())
		self.target_q_function.set_weights(self.q_function.get_weights())

	def predict(self,state):
		state = state[np.newaxis,:]

		noise = self.noise_generator()

		pi = self.policy(state)

		a = pi + noise

		a = a[0]

		return a

	def append_data(self, state, action, reward, new_state):
		self.buffer.store(state, action, reward, new_state)

	def compute_policy_loss(self, state):
		pi = self.policy(state)
		q1,q2 = self.q_function(state, pi)
		return -tf.reduce_mean(q1)

	def compute_critic_loss(self,state,action,reward,new_state):
		pi_targ = self.target_policy(new_state)
		noise = self.noise_generator()
		noise_clip = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
		noise_clip = tf.cast(noise_clip,pi_targ.dtype)
		a = pi_targ + noise_clip
		q1_targ, q2_targ = self.target_q_function(new_state, a)
		q_min = tf.math.minimum(q1_targ, q2_targ)
		y = reward + self.gamma * q_min

		q1, q2 = self.q_function(state,action)

		loss_q1 = tf.reduce_mean((q1 - y)**2)

		loss_q2 = tf.reduce_mean((q2 - y)**2)

		loss = loss_q1 + loss_q2

		return loss

	def soft_update(self,src,trg):
		target_wieghts = []
		for target, source in zip(trg.get_weights(), src.get_weights()):
			fa = self.polyak * target + (1 - self.polyak) * source
			target_wieghts.append(fa)

		trg.set_weights(target_wieghts)

	def train(self,step):
		if not self.buffer.is_train():
			return

		state, action, reward, new_state = self.buffer.get_data()

		with tf.GradientTape() as critic_tape:
			critic_loss = self.compute_critic_loss(state,action,reward,new_state)

		critic_grads = critic_tape.gradient(critic_loss, self.q_function.trainable_variables)
		self.q_opt.apply_gradients(zip(critic_grads, self.q_function.trainable_variables))

		self.critic_loss = critic_loss

		if step % self.policy_train_every == 0:
			with tf.GradientTape() as policy_tape:
				policy_loss = self.compute_policy_loss(state)

			policy_grads = policy_tape.gradient(policy_loss, self.policy.trainable_variables)
			self.opt.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

			self.soft_update(self.policy, self.target_policy)
			self.soft_update(self.q_function, self.target_q_function)

			self.policy_loss = policy_loss








