import matplotlib.pyplot as plt
import random
import numpy as np
import math
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
style.use("ggplot")

df = pd.read_csv('pca-data.txt', header=None, sep='\t')
df=df.T
#initial set

K=3
N=df.shape[1]
#for PCA part:
u=df.sum(axis=1)/N
D=df.subtract(u,axis=0)
Comatrix=np.dot(D,D.T)/N
evalue, evector=np.linalg.eig(Comatrix)

#for i in range(3):
#    if evalue[i]*evector[:][i]==np.dot(Comatrix,evector[:][i]):
#        print("true")

#evalue=pd.DataFrame(data=evalue)
#evalue=evalue.T
#evector=pd.DataFrame(data=evector)
newDim=pd.DataFrame(data=evector,columns=evalue)
newDim=newDim.sort_index(axis=1,ascending=False)
trival=len(evalue)-1
newDim.drop(evalue[trival],axis=1,inplace=True)
Z=np.dot(newDim.T,df)
print("directions of the first two principal components:\n",newDim.values)



###FastMap
fm = pd.read_csv('fastmap-data.txt', header=None, sep='\t')
label = pd.read_csv('fastmap-wordlist.txt', header=None, sep='\t')
X=np.unique(fm.iloc[:,0:2])
label.index=X;
fm.columns=['A','B','D']
###finde the longest pair , find the axis-X
lIndex=np.argmax(fm.iloc[:,2], axis=0)
###3,10
pa=fm.iloc[lIndex,0]
pb=fm.iloc[lIndex,1]
X=np.unique(fm.iloc[:,0:2])
temp=np.zeros([1,len(X)]);
Da=pd.DataFrame(data=temp,columns=X)
Db=pd.DataFrame(data=temp,columns=X)
Dab=fm.iloc[lIndex,2]
for i in X:
    if i< pa:
        Da[i]=fm.ix[(fm.A==i) & (fm.B==pa),'D'].values[0]
    elif i>pa:
        Da[i] = fm.ix[(fm.A == pa) & (fm.B == i), 'D'].values[0]
    if i< pb:
        Db[i]=fm.ix[(fm.A==i) & (fm.B==pb),'D'].values[0]
    elif i>pb:
        Db[i] = fm.ix[(fm.A == pb) & (fm.B == i), 'D'].values[0]
Xi=(Da**2 + Dab**2 -Db**2)/(2*Dab)
### create new Distance matrix
Newfm=pd.DataFrame(columns=['A','B','D'])
for i in range(len(X)-1): #do not include the last word
   if X[i] != pa: #can not be pa
       for j in range(i+1,len(X)):
           if X[j] !=pa:
               Dij=(fm.ix[(fm.A==X[i]) & (fm.B==X[j]),'D'].values[0])**2-(Xi[X[i]]-Xi[X[j]])**2
               Dij=math.sqrt(Dij)
               arr=[X[i],X[j],Dij]
               arr=np.reshape(arr,[1,3])
               temp=pd.DataFrame(data=arr,columns=['A','B','D'])
               Newfm=Newfm.append(temp,ignore_index=True)
#delete label:

#axisX=Xi.drop(pa)
print("the first distance (X-axis) for each point is :\n", Xi.values)
axisX=Xi
axisX.drop([pa],axis=1,inplace=True)
label.drop(pa,inplace=True)


# the second recursive, find the axis-Y
fm=Newfm
lIndex=np.argmax(fm.iloc[:,2], axis=0)
###5,7
pa=fm.iloc[lIndex,0]
pb=fm.iloc[lIndex,1]

X=np.unique(fm.iloc[:,0:2])
temp=np.zeros([1,len(X)]);
Da=pd.DataFrame(data=temp,columns=X)

Dab=fm.iloc[lIndex,2]


for i in X:
    if i< pa:
        Da[i] =fm.ix[(fm.A==i) & (fm.B==pa),'D'].values[0]
    elif i>pa:
        Da[i]= fm.ix[(fm.A == pa) & (fm.B == i), 'D'].values[0]
temp = np.zeros([1, len(X)]);
Db = pd.DataFrame(data=temp, columns=X)
for i in X:
   if i< pb:
        Db[i] =fm.ix[(fm.A==i) & (fm.B==pb),'D'].values[0]
   elif i>pb:
        Db[i] = fm.ix[(fm.A == pb) & (fm.B == i), 'D'].values[0]


Xi=(Da**2 + Dab**2 -Db**2)/(2*Dab)

### create new Distance matrix
Newfm=pd.DataFrame(columns=['A','B','D'])
for i in range(len(X)-1): #do not include the last word
   if X[i] != pa: #can not be pa
       for j in range(i+1,len(X)):
           if X[j] !=pa:
               Dij=(fm.ix[(fm.A==X[i]) & (fm.B==X[j]),'D'].values[0])**2-(Xi[X[i]]-Xi[X[j]])**2
               Dij=math.sqrt(Dij)
               arr=[X[i],X[j],Dij]
               arr=np.reshape(arr,[1,3])
               temp=pd.DataFrame(data=arr,columns=['A','B','D'])
               Newfm=Newfm.append(temp,ignore_index=True)

#got the axis-Y for each word
print("the second distance (Y-axis) for each point is :\n", Xi.values)
axisY=Xi


# plot the points
labels=label.values

#plt.subplots_adjust(bottom = 0.1)
plt.subplots_adjust(bottom = 0.2)
plt.scatter(axisX, axisY, marker='o', cmap=plt.get_cmap('Spectral'))
axisX=axisX.T.values
axisY=axisY.T.values

#for lab, x, y in zip(labels, axisX, axisY):
for label, x, y in zip(labels, axisX, axisY):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-1, 1),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.show()