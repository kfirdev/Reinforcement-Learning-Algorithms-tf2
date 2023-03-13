# create normal network and copy of it as target network
#y_ddqn = reward + self.gamma*self.q_target_model.predict( np.argmax( model.predict(new_state) ) )*(1 - int(done)) )
#append data in memory if memory > batch_size delete memory ( memory = [])
# every n steps update target network wieghts with normal network
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
  except RuntimeError as e:
    print(e)

class ReplayMem(object):
    def __init__(self,mem_size):
        self.mem_size = mem_size
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.new_state_memory = []
        self.terminal_state_mem = []


    def store_data(self,state,action,reward,new_state,done):
        if len(self.state_memory) >= self.mem_size:
            self.state_memory = []
            self.action_memory = []
            self.reward_memory = []
            self.new_state_memory = []
            self.terminal_state_mem = []
            time.sleep(20)

        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.new_state_memory.append(new_state)
        self.terminal_state_mem.append(done)

    def get_memory_current_size(self):
        return len(self.state_memory)

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
    def __init__(self,state_dims,n_actions,epsilon=1.0,epsilon_dec=0.996,epsilon_min=0.01,gamma=0.99, cnn=False):
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.memory = ReplayMem(mem_size=40000)
        
        if cnn:
            self.q_model = self.make_cnn()
        else:
            self.q_model = self.make_model()

        self.q_model_target = self.q_model

        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.batch = 64
        self.fgh = 0


    def store(self,state,action,reward,new_state,done):
        self.memory.store_data(state,action,reward,new_state,done)

    def make_model(self):
        model = Sequential()
        model.add(Dense(256,activation='relu'))
        model.add(Dense(256,activation='relu'))
        model.add(Dense(self.n_actions))

        model.compile(optimizer=Adam(lr=0.0005),loss='mse')
        return model

    def make_cnn(self):
        model = Sequential()
        model.add(Conv2D(64,(3,3) ,activation='relu'))
        model.add(Conv2D(64,(3,3) ,activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.n_actions))

        model.compile(optimizer=Adam(lr=0.0005),loss='mse')
        return model

    def get_action(self,state):
        #print(state)
        state = state[np.newaxis,:]
        if np.random.random() > self.epsilon:
            action_probs = self.q_model.predict(state)
            #print('sdfsdfsdfsdfsfd',action_probs)
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(self.action_space)
            #print('sdfsdfsdfsdfsfd',action)

        return action

    def save_wieghts(self):
        self.q_model.save('wieghts/')

    def load_model(self):
        self.q_model = tf.keras.models.load_model('wieghts/')


    def train(self, episode):
        if self.memory.sample_buffer(self.batch) == 0:
            return
        states, new_states, rewards, terminal, actions = self.memory.sample_buffer(self.batch)
        actions = np.array(actions)
        targets_list = []
        states = np.array(states)
        new_states = np.array(new_states)
        q_current_states = self.q_model.predict(states)
        q_nexts_list = self.q_model.predict(new_states)
        q1 = self.q_model.predict(new_states)
        q2 = self.q_model_target.predict(new_states)

        for df in range(len(new_states)):
            new_state = np.array([new_states[df]])
            reward = np.array(rewards[df])
            done = np.array(terminal[df])
            state_c = np.array([states[df]])
            act = np.array(actions[df])
            state_preds = np.array(q_current_states[df])
            q2_current = np.array(q2[df])
            q1_current = np.array(q1[df])
            #print(state_preds)
            #state_preds = self.q_model.predict(state_c)
            #q_next = self.q_model.predict(new_state)
            #q_next = np.array([q_nexts_list[df]])
            #target = reward + self.gamma*np.max(q_next, axis=1)*(1 - int(done))
            print(q2_current.shape)
            target = reward + self.gamma*q2_current[np.argmax(q1_current)]*(1 - int(done))
            #state_preds = state_preds[0]

            state_preds[act] = target
            #print(state_preds)
            targets_list.append(state_preds)


        targets_list = np.array(targets_list)

        self.q_model.fit(states, targets_list,verbose=0)
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
        if episode % 3 == 0:
            if self.fgh == 0:
                weights = self.q_model.get_weights()
                self.q_model_target.set_weights(weights)
            self.fgh += 1
        else:
            self.fgh = 0










































#
