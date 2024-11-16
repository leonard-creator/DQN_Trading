import numpy as np
import math

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key, test=False):
	data_vec = []
	if test:
		lines = open("test_data/" + key, "r").read().splitlines()
	else:
		lines = open("train_data/" + key, "r").read().splitlines()
	# extract data-values Close,Volume,ROC12,MFI14,FVolatility from line (len=5)	
	data_vec = [[float(elem)] for line in lines[1:] for elem in line.split(",")[4:]]
	data_vec = np.asarray(data_vec, dtype=np.float64)
	
	# reshaping into a np.matrix (rows, data) == (days, values)
	return np.reshape(data_vec, (len(data_vec)//5, 5))

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# returns an an n-day state representation ending at time t
# data is np.ndarray(days, values )  with values =5 
# num_len = 5 is the lenght of one day "tuple" containing 5 measurements from the stock at one day
def getState(data, t, n):
	#import pdb; pdb.set_trace()
	num_len = 5
	d = t - n + 1
	if d >= 0:
		block = data[d:(t +1)]
	else:
		# pad with 0
		#pad = np.zeros((-d*num_len, num_len))
		pad = np.zeros((-d, num_len))
		#pad = np.reshape(pad,(-d,num_len))
		block = np.concatenate((pad, data[0:(t + 1)]))
		#Calculate the differences and apply sigmoid
	
	# calculate differences between measurements along axis 0 (per day diff)

	differences = np.diff(block, axis=0)
	#result = np.concatenate((differences, half2), axis=1)

	# standardize 
	# result = sigmoid(differences)
	return differences
