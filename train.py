from agent.agent import Agent
from keras.models import load_model
from functions import *
import sys
import os
import wandb

if not (len(sys.argv) == 5 or len(sys.argv) == 6):
	print ("Usage: python train.py [stock] [window] [episodes] [model-name] [retrain (optional)]")
	exit()

stock_name, window_size, episode_count, model_name = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),  str(sys.argv[4])

# retrain a model
if len(sys.argv) >5:
	print(f'continue training model {model_name} on stock {stock_name} ...')
	path = "D:\\Dokumente\\UNI\\Master Leiden\\Own_Projects\\DQN_Trader\\models"
	absolute_path = os.path.join(path, model_name)	
	model = load_model(absolute_path)
	window_size = model.layers[0].input.shape.as_list()[1]
	agent = Agent(window_size, model_name=model_name, is_eval=True)
	print(f'Agent with memory:{agent.memory}, name:{model_name}, gamma:{agent.gamma}, lr: {agent.learning_rate}, epsilon:{agent.epsilon}, epsilon decay:{agent.epsilon_decay}, epsilon min:{agent.epsilon_min}')
	model_name = model_name + "_"

# create new model 
else:
	agent = Agent(window_size, model_name=model_name)

data = getStockDataVec(stock_name)
l = len(data) - 1
replay_mem_batch_size = 32 # 32

# initialize Wandb for tracking
run = wandb.init(
    # Set the project where this run will be logged
    project="Deep Q-learning trader",
    # Track hyperparameters and run metadata
    config={
        "model_name": model_name,
		"stock_name": stock_name,
        "window_size": window_size,
		"episode_count": episode_count,
		"memory_buffer_size":len(agent.memory),
		"memory_size": replay_mem_batch_size,
		"lr": agent.learning_rate,
		"gamma": agent.gamma,
		"epsilon": agent.epsilon,
		"epsilon_decay": agent.epsilon_decay,
		"target_N": True,
		"NN_architecture":None,
		"reward_type":"profit_based", # reward strategy

    },
)



for episode in range(episode_count + 1):
	print("Episode " + str(episode) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for timestep in range(l):
		#import pdb; pdb.set_trace()
		action = agent.act(state)

		# hold
		next_state = getState(data, timestep + 1, window_size + 1)
		reward = 0

		# buy
		if action == 1:
			#if len(agent.inventory)<=2:
			#	reward = 1
			agent.inventory.append(data[timestep])
			##print ("Buy: " + formatPrice(data[timestep]))
			
		
		# sell first 
		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = data[timestep] - bought_price # why not reward -1 for punishing? doesnt seem to work though

			total_profit += data[timestep] - bought_price
			##print( "Sell first: " + formatPrice(data[timestep]) + " | Profit: " + formatPrice(data[timestep] - bought_price))
		
		# sell last
		elif action == 3 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop() # pop last in inventory
			reward = data[timestep] - bought_price # why not reward -1 for punishing? doesnt seem to work though

			total_profit += data[timestep] - bought_price
			##print( "Sell last: " + formatPrice(data[timestep]) + " | Profit: " + formatPrice(data[timestep] - bought_price))

		# updating end of timesteps
		if timestep == l - 1:
			done = True 
		else:
			done = False
		# append state to memory buffer 
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		# update target model with new weights
		if timestep % 100 == 0:
			agent.target_model.set_weights(agent.model.get_weights())

		# update total profit every 100 timesteps
		if timestep % 25 == 0:
			wandb.log({"total_profit":total_profit, "inventory":len(agent.inventory),
			   "timestep":timestep,"episode":episode, "epsilon":agent.epsilon, "buffer_size":len(agent.memory)})

		
		# Experienced Replay step
		if len(agent.memory) > replay_mem_batch_size:
			agent.expReplay(replay_mem_batch_size)

		if done:
			print( "--------------------------------")
			print ("Total Profit: " + formatPrice(total_profit))
			print( "--------------------------------")
			# log last metrics before new episode begins
			#wandb.log({"total_profit":total_profit, "inventory":len(agent.inventory),
			#   "timestep":timestep,"episode":episode, "epsilon":agent.epsilon, "buffer_size":len(agent.memory)})

	# intermediate saving
	if episode % 1 == 0:
		agent.model.save("models/" + agent.model_name + str(episode))
