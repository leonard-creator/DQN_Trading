import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent:
	def __init__(self, state_size, model_name, is_eval=False, use_target=True):
		self.state_size = state_size # normalized previous days
		self.action_size = 4 # sit, buy, sell first, sell last
		self.memory = deque(maxlen=1000) # 1000 change to sampling out of the last 100
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.999 #0.95
		self.epsilon = 0.99 #1.0 # best model reward_target8 on ^GSPC 0.9
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.is_eval = is_eval
		if self.is_eval:
			self.model = load_model("models/" + model_name)
		else:
			self.model = self._model()
		if use_target:
			self.target_model = self.copy_network(self.model)

	def _model(self):
		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		#model.add(BatchNormalization())
		model.add(Dense(units=128, activation="relu"))
		#model.add(BatchNormalization())
		model.add(Dense(units=128, activation="relu"))
		#model.add(BatchNormalization())
		model.add(Dense(units=64, activation="relu"))
		#model.add(BatchNormalization())
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(learning_rate= self.learning_rate))

		return model

	def copy_network(self , model):
		model_copy = keras.models.clone_model(model)
		model_copy.build((None, self.state_size))
		model_copy.compile(loss="mse",	
                      optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=["mae"])
		model_copy.set_weights(model.get_weights())
		
		return model_copy
	
	def create_target(self, state, action, reward, state_next, done):
		""" 
        Function to calculate the target with double DQN, 
        target-network or without target network and returning the updated estimated Q-values.
        """
		target = reward
		if self.double_clipped:
			if not done:
				# Bellman Equation
				target = reward + self.gamma * min(self.predict(state_next, self.q_target_net)[0][action], 
                                                   self.predict(state_next, self.q_net)[0][action])
            # using the DQN-agents predict function to estimate the Q-values 
			target_f = self.predict(state, self.q_target_net)
		elif self.use_target:
			if not done:
				# Bellman Equation
				target = reward + self.gamma * np.amax(self.predict(state_next, self.q_target_net)[0])
			# using the DQN-agents predict function to estimate the Q-values
			target_f = self.predict(state, self.q_target_net)
		else:
			if not done:
				# Bellman Equation
				target = reward + self.gamma * np.amax(self.predict(state_next, self.q_net)[0])
			# using the DQN-agents predict function to estimate the Q-values
			target_f = self.predict(state, self.q_net)
		target_f[0][action] = target
		
		return target_f

	def act(self, state):
		# Use a single condition for exploration vs. exploitation
		if not self.is_eval and np.random.uniform(0, 1) <= self.epsilon:
			# Random action for exploration
			return random.randint(0, self.action_size - 1)  # More efficient than randrange
		
		options = self.model(state, training=False).numpy()  # Avoid unnecessary gradient tracking		return np.argmax(options[0])
		return int(np.argmax(options[0]))  # Directly return as an int for efficiency

	def expReplay(self, batch_size):
		#import pdb;pdb.set_trace()
		# Randomly sample batch_size indices from the memory array
		#sampled_indices = random.sample(range(l), batch_size)
		random_idxs = np.random.choice(len(self.memory), batch_size, replace=False )
		# Use the sampled indices to append to the mini_batch
		#for i in sampled_indices:
		#	mini_batch.append(self.memory[i])
		

		#for state, action, reward, next_state, done in mini_batch:
		for batch_idx in random_idxs:
			# get the values
			state, action, reward, next_state, done = self.memory[batch_idx]
			if done:
				target = reward
			else:
				# temporal difference learning TD-target
				#target = reward + self.gamma * np.amax(self.model(next_state)[0]) # model.predict
				# using target model for prediction
				target = reward + self.gamma * np.amax(self.target_model(next_state, training=False)[0]) 

			target_f = self.model(state, training=False).numpy() # model.predict
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 
