import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

f=open('./CSR1.DAT')
trainscale=300
data=[]
flag=0
for line in f.readlines():
    str = line.strip().split("\t")
    tmp = []
    if str[0]=='1':
        flag=1
    if line=='\n':
        flag=0
    if flag==1:
        for i in range(1, len(str)):
            tmp.append(float(str[i]))
        data.append(tmp)

min=min(data)[0]
max=max(data)[0]
scale=max-min


for num in data:
    num[0]=(num[0]-min)/1#scale


ytrain=np.array(data[0:trainscale])
ytest=np.array(data[trainscale:len(data)])

ytrain=ytrain.ravel()
ytest=ytest.ravel()

y=np.array(data)
y=y.ravel()


norm=1
x = np.arange(1/norm, (len(data)+1)/norm, 1/norm)
xtrain=np.arange(1/norm, (trainscale+1)/norm, 1/norm)
xtrain.shape=(-1,1)
#xpredict = np.arange(trainscale+1, len(data)+1, 1)
xpredict = np.arange(1/norm, (len(data)+1)/norm, 1/norm)
xpredict.shape=(-1,1)

xtest = np.arange((trainscale+1)/norm,( len(data)+1)/norm, 1/norm)
xtest.shape=(-1,1)





train=svm.SVR(kernel='rbf', C=1e3, gamma=20)
train.fit(xtrain,ytrain)
ypredict=train.predict(xpredict)



plt.figure(figsize=(21, 8))
plt.plot(xtest, ytest, label="test", color='g')
plt.plot(xpredict, ypredict, label="predict", color='r')
plt.plot(x, data, label="raw", color='b')
plt.legend()
plt.title("SVR")
plt.show()




"""
x = np.arange(1, len(data)+1, 1)
x.reshape(-1,1)
x.tolist()
print(x)

x=[]
for i in range(1,(len(data)+1)):
    x.append(i)

train=svm.SVR()
train.fit(x,data)
print(train.predict([[1]]))

plt.figure(figsize=(21, 8))
plt.plot(x, data, label="CDF", color='g')
plt.title("CDF")
plt.show()

print(data)
print(x)
"""