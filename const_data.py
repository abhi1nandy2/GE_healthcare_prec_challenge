import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('rrest-syn001_data.csv', usecols = [1], header = None)
arr_list = []
# with open('ark.csv') as f:
for i in range(50):
	offset = 
	data = np.array(df.iloc[:,0])
	# print(data)
	t_inv = data[200:275]
	# exit()
	# plt.plot(df.iloc[:,0])
	# plt.show()

	base = np.mean(data[0:375])
	data[200:275] = 2*base - t_inv
	arr_list.extend((data[0:375] + offset).tolist())
	# f=open('ark.csv')
		
df = pd.DataFrame(columns= ['abnormal'])
df['abnormal'] = arr_list
df.to_csv('ark.csv', index=False)
# plt.plot(df.iloc[:,0])
# plt.show()
# plt.plot(data[0:375])
# plt.show()