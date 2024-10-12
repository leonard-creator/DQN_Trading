from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 5:
	print ("Usage: python train.py [stock] [window] [episodes] [model-name]")
	exit()

stock_name, window_size, episode_count, model_name = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),  str(sys.argv[4])

agent = Agent(window_size, model_name=model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32 # 32

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
			print ("Buy: " + formatPrice(data[timestep]))
			#reward = -1
		
		# sell first 
		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = data[timestep] - bought_price # why not reward -1 for punishing? doesnt seem to work though

			total_profit += data[timestep] - bought_price
			print( "Sell first: " + formatPrice(data[timestep]) + " | Profit: " + formatPrice(data[timestep] - bought_price))
		
		# sell last
		elif action == 3 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop() # pop last in inventory
			reward = data[timestep] - bought_price # why not reward -1 for punishing? doesnt seem to work though

			total_profit += data[timestep] - bought_price
			print( "Sell last: " + formatPrice(data[timestep]) + " | Profit: " + formatPrice(data[timestep] - bought_price))

		# updating end of timesteps
		if timestep == l - 1:
			done = True 
		else:
			done = False

		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		# update target model with new weights
		if timestep % 100 == 0:
			agent.target_model.set_weights(agent.model.get_weights())

		if done:
			print( "--------------------------------")
			print ("Total Profit: " + formatPrice(total_profit))
			print( "--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)
	if episode % 1 == 0:
		agent.model.save("models/" + agent.model_name + str(episode))
