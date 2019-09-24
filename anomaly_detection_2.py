import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
from scipy.stats import norm
from math import ceil

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LSTM, ConvLSTM2D, Concatenate, Merge
from keras.models import Model
from keras import backend as K
from keras.layers import BatchNormalization, Activation
from keras.optimizers import Adam, Adadelta
from keras.layers import add, Dropout
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os
fig, ax = plt.subplots(1, 1)
mean, var, skew, kurt = norm.stats(moments='mvsk')

lr = 0.001
factor = 0.9
patience = 5
min_lr = 0.0001
epochs = 30
noise = 0
dropout = 0.2
batch_size = 16

WL = int(sys.argv[2])
stride = int(sys.argv[4])

data = pd.read_csv(str(sys.argv[1]), header = None, usecols = [0,1]).iloc[0:10000]
# print(total_data.shape)
# exit()
# plt.plot(total_data.iloc[0:1000,0])
# plt.savefig('normal.jpg')
# plt.show()
# plt.close()
# plt.plot(total_data.iloc[2000:6000,0])
# plt.plot(total_data.iloc[4000:4300,0], 'r')
# plt.savefig('abnormal_1.jpg')
# plt.show()
# plt.close()
# plt.plot(total_data.iloc[7700:12700,0])
# plt.plot(total_data.iloc[10000:10600,0], 'g')
# # plt.savefig('abnormal_2.jpg')
# # plt.show()
# # plt.plot(total_data.iloc[10900:12500,0])
# plt.plot(total_data.iloc[10800:11300,0], 'y')
# plt.savefig('abnormal_2_3.jpg')
# plt.show()
# plt.close()
# exit()
# data = total_data.iloc[np.r_[0:12000-96:1, 12672-96:353*96:1, 360 * 96:86 * 96:1, 93 * 96:total_data.shape[0]:1]]
# data = total_data.iloc[np.r_[0:4000:1, 4500:total_data.shape[0]:1]]
# data = total_data.iloc[np.r_[0:4000:1, 4500:10000:1, 10600:10900:1, 11500:total_data.shape[0]:1]]
# data = data.reset_index(drop = True)
# anom_data = total_data.iloc[4000:4500]
total_data = pd.read_csv('mitdbx_mitdbx_108.txt', header = None, usecols = [2], delimiter = '\t')
anom_data = total_data.iloc[np.r_[4000:4500:1, 10000:10600:1, 10900: 11500:1]]
anom_data_2 = anom_data.reset_index(drop = True)
anom_data = np.zeros((anom_data_2.shape[0],2))
anom_data[:,1]=np.array(anom_data_2).reshape(-1)
anom_data[:,0] = -data.iloc[0:anom_data_2.shape[0],0]
# print(data.shape)
print(anom_data.shape)
# data = pd.read_csv('rrest-syn001_data.csv', usecols = [1])
i=0
while(i*stride+WL <= data.shape[0]):
	i=i+1
no_of_wins = i
print('no. of wins')
print(no_of_wins)
# exit()
train_data = data.iloc[0:int(0.9 * data.shape[0])]
val_data = data.iloc[int(0.9 * data.shape[0]):]
train_data = train_data.reset_index(drop = True)
val_data = val_data.reset_index(drop = True)
print(np.shape(val_data))
# exit()
scaler = StandardScaler()
scaler.fit(train_data)

# labels = np.load(str(sys.argv[2]))
# train_labels = labels[0:int(0.8 * no_of_wins)]
# val_labels = labels[int(0.8 * no_of_wins):]
# print(np.shape(train_labels))
# print(np.shape(val_labels))

# print(data.iloc[0:WL].shape)
# exit()
train_frames = []
val_frames = []
anom_frames = []

# for i in range(int(train_data.shape[0]/WL)):
i = 0
while(i*stride+WL <= train_data.shape[0]):
	train_frames.append(np.transpose(scaler.transform(train_data.iloc[(i * stride):(i * stride) + WL])))
	i = i+1

print(len(train_frames))
i=0
while(i * stride + WL <= val_data.shape[0]):
	# win_no = i + len(train_frames)
	val_frames.append(np.transpose(scaler.transform(val_data.iloc[(i * stride):(i * stride) + WL])))
	i = i+1
print(len(val_frames))
i=0
while(i * stride + WL <= anom_data.shape[0]):
	anom_frames.append(np.transpose(scaler.transform(anom_data[(i * stride):(i * stride) + WL])))
	i=i+1
train_arr = np.array(train_frames)
train_arr = train_arr.reshape((train_arr.shape[0], train_arr.shape[1], train_arr.shape[2], 1))
val_arr = np.array(val_frames)
val_arr = val_arr.reshape((val_arr.shape[0], val_arr.shape[1], val_arr.shape[2], 1))
anom_arr = np.array(anom_frames)
print(np.shape(anom_arr))
anom_arr = anom_arr.reshape((anom_arr.shape[0], anom_arr.shape[1], anom_arr.shape[2], 1))

print(np.shape(train_arr))
print(np.shape(val_arr))
print(np.shape(anom_arr))

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor,
                              patience=patience, min_lr=min_lr, verbose = 1)

input_shape = (train_arr.shape[1], train_arr.shape[2], 1)

def bn_var_conv_block(dropout,x, batch_norm, filter_sizes, no_of_filters, scale, residual = False):
    res_inp = x
    if batch_norm == True:
        x = BatchNormalization(scale = scale)(x)
    y = Conv2D(no_of_filters, (1, filter_sizes),kernel_initializer=keras.initializers.glorot_uniform(seed = 1), padding = 'same')(x)
    
    y = Activation('relu')(y)
    y = MaxPooling2D((1, 2), padding = 'same')(y)
    y = Dropout(rate = dropout, seed = 1)(y)
    if residual == True:
        y = Conv2D(x.get_shape().as_list()[3], (1, filter_sizes), activation='relu',kernel_initializer=keras.initializers.glorot_uniform(seed = 1), padding = 'same')(y)
        temp = MaxPooling2D((1, 2), padding = 'same')(res_inp)
        #temp = BatchNormalization(scale = scale)(temp)
        y = add([y, temp])
    return(y)

def bn_var_deconv_block(dropout,x, filter_sizes, no_of_filters, scale, residual = False):
    res_inp = x
    y = BatchNormalization(scale=scale)(x)
    y = Conv2D(no_of_filters, (1, filter_sizes),kernel_initializer=keras.initializers.glorot_uniform(seed = 1), padding = 'same')(y)
    
    y = Activation('relu')(y)
    y = UpSampling2D((1, 2))(y)
    y = Dropout(rate = dropout, seed = 1)(y)    
    if residual == True:
        y = Conv2D(x.get_shape().as_list()[3], (1, filter_sizes),kernel_initializer=keras.initializers.glorot_uniform(seed = 1), activation='relu', padding = 'same')(y)
        temp = UpSampling2D((1, 2))(res_inp)
        #temp = BatchNormalization(scale = scale)(temp)
        y = add([y, temp])
    return(y)


def bn_var_create_cnn(lr, input_shape, conv_filter_sizes, conv_no_of_filters, deconv_filter_sizes, deconv_no_of_filters, last_layer_size, dropout, noise):
    input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format
    #temp = input_img
    #input_image = Input(shape=(input_shape)) #to avoid any possible change in input layer
    if noise == 1:
        y = keras.layers.GaussianNoise(0.0001)(input_img)
    else:
        y = input_img
    for i in range(len(conv_filter_sizes)):
        if i==0:
            temp = bn_var_conv_block(dropout,y,False, conv_filter_sizes[i], conv_no_of_filters[i], scale = False, residual = False)
        else:
            temp = bn_var_conv_block(dropout,temp, True,conv_filter_sizes[i], conv_no_of_filters[i], scale = False, residual = True)
    for i in range(len(deconv_filter_sizes)):
        if i == len(deconv_filter_sizes) - 1: #Last layer (useful, if batch_norm becomes the last layer before non-linear layer)
            temp = bn_var_deconv_block(dropout,temp, deconv_filter_sizes[i], deconv_no_of_filters[i], scale = False, residual = False)
        else:
            temp = bn_var_deconv_block(dropout,temp, deconv_filter_sizes[i], deconv_no_of_filters[i], scale = False, residual = True)
    #bring it to the desired shape in the above step
    decoded = Conv2D(1, (1, last_layer_size))(temp)

    autoencoder = Model(input_img, decoded)
    optimizer = Adam(lr = lr)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    return autoencoder

def train_model(model, filepath, log_filepath, train, output, valid, valid_output, epochs, batch_size):
    chkpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    csv_logger = CSVLogger(log_filepath)
    model.fit(train, output, validation_data = (valid, valid_output), epochs=epochs, verbose=1, batch_size=batch_size, callbacks=[reduce_lr, chkpt, csv_logger])

conv_filter_sizes = [int(0.8 * WL), int(0.8 * WL)]
deconv_filter_sizes = conv_filter_sizes[::-1]
conv_no_of_filters = [32, 32]
deconv_no_of_filters = [32, 32]

last_filter_size = ceil(ceil(input_shape[1]/2)/2) * 4 - input_shape[1] + 1

model = bn_var_create_cnn(lr,input_shape, conv_filter_sizes, conv_no_of_filters, deconv_filter_sizes, deconv_no_of_filters, last_filter_size, dropout, noise)

best_model = 'best_model_new.h5'

with open('model_report.txt','w') as fh:
# Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

training = sys.argv[3]
if training == 'train':
	train_model(model, best_model, 'anomaly_log.log', train_arr, train_arr, val_arr, val_arr, epochs, batch_size)

print('Finished training\n')
print('Loading best model')

model = load_model(best_model)

train_err = []
num_str = int(WL/stride)
for i in range(0,len(train_frames)-len(train_frames)%num_str,num_str):
	sum = 0
	for j in range(0,num_str):
		temp = i+j
		output = model.predict(train_arr[temp].reshape((1, train_arr.shape[1], train_arr.shape[2], 1)), batch_size = 1).reshape((train_arr.shape[1], train_arr.shape[2], 1))
		sum+=np.mean((output - train_arr[temp])**2)
	train_err.append(sum/num_str)

val_err = []
for i in range(0,len(val_frames)-len(val_frames)%num_str,num_str):
	sum = 0
	
	for j in range(0,num_str):
		temp = i+j
		output = model.predict(val_arr[temp].reshape((1, train_arr.shape[1], train_arr.shape[2], 1)), batch_size = 1).reshape((train_arr.shape[1], train_arr.shape[2], 1))
		sum += (np.mean((output - val_arr[temp])**2))
	val_err.append(sum/num_str)
anom_err = []
count = 0
for i in range(0,len(anom_frames)-len(anom_frames)%num_str,num_str):
	sum = 0
	for j in range(0,num_str):
		temp = i+j
		output = model.predict(anom_arr[temp].reshape((1, train_arr.shape[1], train_arr.shape[2], 1)), batch_size = 1).reshape((train_arr.shape[1], train_arr.shape[2], 1))
		sum += np.mean((output - anom_arr[temp])**2)

	# if (sum/num_str)> 0.05:
	# 	arr_list = []
	# 	for j in range(0,num_str-1):
	# 		temp = i+j
	# 		arr_list.extend(anom_arr[temp].reshape(-1)[0:stride].tolist())
	# 	arr_list.extend(anom_arr[temp].reshape(-1).tolist())
	# 	df = pd.DataFrame(columns = ['data'])
	# 	df['data'] = arr_list
	# 	df.to_csv('./anom_frames/' + str(count) + '.csv', index = False)
	# 	count = count+1
		# for j in range(0,num_str):
		# 	temp = i+j
		# 	df = pd.DataFrame(columns = ['data'])
		# 	df['data'] = anom_arr[temp].tolist()
		# 	df.to_csv('./anom_frames/' + str(count) + '.csv', index = False)
		# 	count = count+1

	anom_err.append(sum/num_str)
train_err_arr = np.array(train_err)
val_err_arr = np.array(val_err)
anom_err_arr = np.array(anom_err)
train_err_arr = np.sort(train_err_arr)
val_err_arr = np.sort(val_err_arr)
anom_err_arr = np.sort(anom_err_arr)
ax.scatter(train_err_arr, norm.pdf(train_err_arr, loc = np.mean(train_err_arr), scale = np.std(train_err_arr)))#,'r-', lw=5, alpha=0.6, label='norm pdf')
ax.scatter(val_err_arr, norm.pdf(val_err_arr, loc = np.mean(train_err_arr), scale = np.std(train_err_arr)), color = 'g')
ax.scatter(anom_err_arr, norm.pdf(anom_err_arr, loc = np.mean(train_err_arr), scale = np.std(train_err_arr)), color = 'k')
train_err_mean = np.mean(train_err_arr)
train_err_std = np.std(train_err_arr)
plt.scatter([train_err_mean - 2 * train_err_std, train_err_mean - 3 * train_err_std, train_err_mean + 2 * train_err_std, train_err_mean + 3 * train_err_std],[0,0,0,0], color = 'r')
plt.show()