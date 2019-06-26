
#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
'''
pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_seq_items',None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)
'''

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
    
#colnames = ['X_cordinate','Y_coordinate','Distance','Direction_angle','Activity_detection','Activity']
#dataset = pd.read_csv('walking.csv', names=colnames)
#dataset = read_csv('walking.csv', header=0, index_col=0)
#dataset = read_csv('walking3.csv',header=None,names=colnames)
dataset = read_csv('walking3.csv',header=0)
print(dataset)
#dataset.drop(dataset.columns[[3]], axis=1, inplace=True)
#print(dataset)

#dataset.drop(dataset.columns[[4]], axis=1, inplace=True)

values = dataset.values
print(values)


# integer encode direction
encoder = LabelEncoder()
#print(encoder)
values[:,4] = encoder.fit_transform(values[:,4])
print('encoded',values)
print('encoded[4]',values[:,4]) 
values12=(encoder.inverse_transform(list(values[:,4])))
print('decoded',values12)


#convert into one hot encoder to avoid confusion
#onehotencoder=OneHotEncoder(categorical_features=[4])
#values=onehotencoder.fit_transform(values).toarray()
#print(values)

# ensure all data is float
#values = values.astype('float32')
#print(values)

# normalize features  //include number of classes
#scaler = MinMaxScaler(feature_range=(0, 1))
#print(scaler)

#scaled = scaler.fit_transform(values)
#print('scaled\n',scaled)

# frame as supervised learning
#reframed = series_to_supervised(scaled, 1, 1)
reframed = series_to_supervised(values, 1, 1)

print(reframed)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[6,7,8,9,11]], axis=1, inplace=True)
print('head\n',reframed.head())  #by default print first 5 rows
# split into train and test sets
values = reframed.values
print('values\n',values)


# for 1st year
n_train_hours = 1 * 24
train = values[:n_train_hours, :]
print('train\n',train)
test = values[n_train_hours:, :]
print('test\n',test)

# split into input and outputs
#train_X, train_y = train[:, :-1], train[:, -1]
train_X, train_y = train[:, :-1], train[:, -1]
print('train_X \n',train_X)
print(' train_y\n', train_y)
#test_X, test_y = test[:, :-1], test[:, -1]

test_X, test_y = test[:, :-1], test[:, -1]
print('test_X \n',test_X)
print(' test_y\n', test_y)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
print('train_X \n',train_X)
print('train_X.shape[0] \n',train_X.shape[0])
print('train_X.shape[1] \n',train_X.shape[1])


test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print('test_X \n',test_X)
print('test_X.shape[0] \n',test_X.shape[0])
print('test_X.shape[1] \n',test_X.shape[1])


print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam') #adam
# fit network

history = model.fit(train_X, train_y, epochs=3, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

print('history\n',history)
# plot history

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# make a prediction
yhat = model.predict(test_X)
print('test_y\n',test_y)
print('yhat \n',yhat )


print('test_X \n',test_X)
print('test_X.shape[0] \n',test_X.shape[0])
print('test_X.shape[1] \n',test_X.shape[1])
print('test_X.shape[2] \n',test_X.shape[2])

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
#print('inv_yhat \n',inv_yhat)

#scaler = MinMaxScaler(feature_range=(0, 1)).fit(inv_yhat)
#inv_yhat = scaler.inverse_transform(inv_yhat)
#print('inv_yhat \n',inv_yhat)

inv_yhat = inv_yhat[:,0]
print('inv_yhat \n',inv_yhat)

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
print('test_y \n',test_y)
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
#print('inv_y \n',inv_y )
#inplace of 2 set number of classes includes 0
#scaler = MinMaxScaler(feature_range=(0, 1)).fit(inv_y)

#inv_y = scaler.inverse_transform(inv_y)
#print('inv_y \n',inv_y )
inv_y = inv_y[:,0]
print('inv_y \n',inv_y )
#converting from integer to category
inv_y12 = list(encoder.inverse_transform(list(inv_y)))
print('inv_y \n',inv_y12 )

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)



