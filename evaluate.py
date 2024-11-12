import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta


if len(sys.argv) != 4:
	print("Usage: python evaluate.py [stock] [model_name] [verbose]")
	exit()

stock_name, model_name, verbose = sys.argv[1], sys.argv[2] , sys.argv[3]
path = "D:\\Dokumente\\UNI\\Master Leiden\\Own_Projects\\DQN_Trader\\models"
absolute_path = os.path.join(path, model_name)
print(absolute_path)
model = load_model(absolute_path)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, model_name=model_name, is_eval=True, use_target=True)
data = getStockDataVec(stock_name, test=True)
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

if verbose == "yes":
	verbose=1
else:
	verbose=0
# graph creation
buy_signals = []
sell_signals = []
start_date = '2011-01-01'
start_date = datetime.strptime(start_date, '%Y-%m-%d')



for t in range(l):
	action = agent.act(state)

	# sit
	next_state = getState(data, t + 1, window_size + 1)
	#update date 
	current_date = start_date + timedelta(days=t)


	if action == 1: # buy
		agent.inventory.append(data[t])
		buy_signals.append((t, data[t]))
		if not verbose: print( "Buy: " + formatPrice(data[t]))
	
	# sell first out of inventory
	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		total_profit += data[t] - bought_price
		sell_signals.append((t, data[t]))
		if not verbose: print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

	# sell last out of inventory 
	elif action == 3 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop()
		total_profit += data[t] - bought_price
		sell_signals.append((t, data[t]))
		if not verbose: print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))


	done = True if t == l - 1 else False
	state = next_state

	if done:
		print ("--------------------------------")
		print (stock_name + " Total Profit: " + formatPrice(total_profit))
		print( "--------------------------------")
		# Create a DataFrame for plotting
		df = pd.DataFrame({'date': [i for i in range(len(data))], 'price': [data[t] for t in range(len(data))] })

		# Plot the price data
		plt.figure(figsize=(10, 5))
		plt.plot(df['date'], df['price'], label='Price')

		# Plot buy signals
		buy_dates, buy_prices = zip(*buy_signals)
		plt.scatter(buy_dates, buy_prices, marker='^', color='g', label='Buy')

		# Plot sell signals
		sell_dates, sell_prices = zip(*sell_signals)
		plt.scatter(sell_dates, sell_prices, marker='v', color='r', label='Sell')
		plt.legend(loc="upper left", title= "Total Profit: "+formatPrice(total_profit))
		plt.xlabel('Count of Days')
		plt.ylabel('Price of stock')
		plt.title(f'{model_name} on stock data {stock_name}')


		plt.show()
