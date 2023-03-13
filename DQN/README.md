# This is an implementation of the DQN algorithm from the paper https://arxiv.org/pdf/1312.5602.pdf

# The model is trying to predict the future discounted reward of each action given a state

# This model is a Discreate action model which means that it can do only a finite number of actions

# Because the model is trying to the future discounted reward of each action we take the argmax of the prediction as our action