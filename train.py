import pandas as pd
import numpy as np
dataset = pd.read_csv("train-data-mdfdd.csv")
dataset.dropna(inplace=True)
z=dataset.iloc[:,0].values
x = dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
labelencoder = LabelEncoder()

sc=StandardScaler()

x[:,0]=labelencoder.fit_transform(x[:,0])
x[:,1]=labelencoder.fit_transform(x[:,1])
x[:,4]=labelencoder.fit_transform(x[:,4])
x[:,5]=labelencoder.fit_transform(x[:,5])
x[:,6]=labelencoder.fit_transform(x[:,6])

#x=sc.fit_transform(x)

one=OneHotEncoder(categories='auto')
x=one.fit_transform(x).toarray()
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.svm import SVR
c=SVR(kernel = "linear",gamma="auto")
c.fit(x_train,y_train)
y_p=c.predict(x_test)
from sklearn import metrics
from sklearn.metrics import r2_score
print(r2_score(y_test,y_p))
print(np.sqrt(metrics.mean_squared_error(y_test,y_p)))

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.scatter(x[:,0],y,z,color="red",marker="o")
ax.scatter(x[:,1],y,z,color="brown")
ax.scatter(x[:,2],y,z,color="black",marker="o")
ax.scatter(x[:,3],y,z,color="yellow",marker="o")
ax.scatter(x[:,4],y,z,color="green",marker="o")
ax.scatter(x[:,5],y,z,color="orange",marker="o")
ax.scatter(x[:,6],y,z,color="blue",marker="o")
ax.scatter(x[:,7],y,z,color="cyan",marker="o")
ax.scatter(x[:,8],y,z,color="olive",marker="o")
ax.scatter(x[:,9],y,z,color="lime",marker="o")
ax.scatter(x[:,10],y,z,color="purple",marker="o")
  
"""
