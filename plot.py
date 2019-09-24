import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('rrest-syn001_data.csv', usecols = [1])
plt.plot(df.iloc[:,0])
plt.show()