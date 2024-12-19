import numpy as np

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

# This estimator scales and translates each feature individually such that it is in the given range 
# on the training set, e.g. between zero and one or -1 and 1
# input: x = 1d ndarray
def MinMaxScale(x, r_min=-1, r_max=0):
	X_std = (x - x.min()) / (x.max() - x.min())
	# return scaled
	return X_std * (r_max - r_min) + r_min 

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
	# only for "Close" and "Volume" in first and second column
	# Close,Volume,ROC12,MFI14,FVolatility
	block = block.T # transpose
	cl_vol = block[:2]
	cl_vol_dif = np.diff(cl_vol)

	# concatenate to one flat vector in O(n), ready for NN input
	##result = np.concatenate((cl_vol_dif.flatten(), block[2:].flatten()))

	# standardize Close, Vol, Roc and MFI
	cl_std = MinMaxScale(cl_vol_dif[0])
	vol_std = MinMaxScale(cl_vol_dif[1])
	roc_std = MinMaxScale(block[2])
	MFI_std = MinMaxScale(block[3])

	# concatenate to flattened vector as result
	# import pdb;pdb.set_trace()
	result = np.concatenate((cl_std, vol_std,roc_std, MFI_std, block[4]), axis=0)


	# standardize 
	# result = sigmoid(differences)
	return result
