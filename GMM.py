#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
#from scipy.io import loadmat
#mat = loadmat('Dataset1.mat',struct_as_record=True)


# In[2]:


# dataset UKM
data_train = pd.read_excel('UKM.xls','Training_Data')
data_test = pd.read_excel('UKM.xls','Test_Data')
label=['very_low','Low','Middle','High']
for i in label:
    data_test.iloc[ : ,5] = data_test.iloc[ : ,5].replace(i,label.index(i)) 
    data_train.iloc[ : ,5] = data_train.iloc[ : ,5].replace(i,label.index(i))
data_train=data_train.values
data_test=data_test.values
c0=[]
c1=[]
c2=[]
c3=[]
for i in range(len(data_train)):
    if data_train[i,5]==0:
        c0.append(data_train[i, :5])
    elif data_train[i,5]==1:
        c1.append(data_train[i, :5])
    elif data_train[i,5]==2:
        c2.append(data_train[i, :5])
    elif data_train[i,5]==3:
        c3.append(data_train[i, :5])    
c=[c0,c1,c2,c3]
classes1=[]
for i in c:
    classes1.append(np.array(i))  
labelU=[0,1,2,3]    


# In[3]:


#iris
dataset=pd.read_csv('iris (1).data')
labels=['Iris-setosa','Iris-versicolor','Iris-virginica']
columns=['A','B','C','D']
for i in labels:
    dataset.iloc[ : ,4] = dataset.iloc[ : ,4].replace(i,labels.index(i)) 
for i in columns:
    dataset[i]=(dataset[i]-dataset[i].mean())/dataset[i].std()
Data=dataset.values
train=np.concatenate((Data[ :40, :5],Data[50:90, :5],Data[100:140, :5]))
test=np.concatenate((Data[40:50, :5],Data[90:100, :5],Data[140:150, :5]))
c0=[]
c1=[]
c2=[]
for i in range(len(train)):
    if train[i,4]==0:
        c0.append(train[i, :4])
    elif train[i,4]==1:
        c1.append(train[i, :4])
    elif train[i,4]==2:
        c2.append(train[i, :4])  
c=[c0,c1,c2]
classes2=[]
for i in c:
    classes2.append(np.array(i))
labelI=[0,1,2]    


# In[4]:


# vehicle
data=pd.read_table('Vehicle.dat', sep=' ')
for i in range(18):
    data.iloc[ : ,i]=(data.iloc[ : ,i]-data.iloc[ : ,i].mean())/data.iloc[ : ,i].std()
l=['van','saab','bus','opel']
for i in l:
    data.iloc[ : ,18] = data.iloc[ : ,18].replace(i,l.index(i)) 
data.drop('Unnamed: 19',axis=1,inplace=True)
data=data.values
datatrain=data[ :74, : ]
datatest=data[74: , : ]
C0=[]
C1=[]
C2=[]
C3=[]
for i in range(len(datatrain)):
    if datatrain[i,18]==0:
        C0.append(datatrain[i, :19])
    elif datatrain[i,18]==1:
        C1.append(datatrain[i, :19])
    elif datatrain[i,18]==2:
        C2.append(datatrain[i, :19])
    elif datatrain[i,18]==3:
        C3.append(datatrain[i, :19])   
C=[C0,C1,C2,C3] # ba label
classes3=[]
for i in C:
    classes3.append(np.array(i))
labelV=[0,1,2,3]    


# In[5]:


# Health
datas=pd.read_table('Health.dat', sep=' ')
for i in range(13):
    datas.iloc[ : ,i]=(datas.iloc[ : ,i]-datas.iloc[ : ,i].mean())/datas.iloc[ : ,i].std()
l=[1,2]
for i in l:
    datas.iloc[ : ,13] = datas.iloc[ : ,13].replace(i,l.index(i)) 
datas=datas.values
data__train=datas[ :215, : ]
data__test=datas[215: , : ]
C0=[]
C1=[]
for i in range(len(data__train)):
    if data__train[i,13]==0:
        C0.append(data__train[i, :13])
    elif data__train[i,13]==1:
        C1.append(data__train[i, :13])
C=[C0,C1] # ba label
classes4=[]
for i in C:
    classes4.append(np.array(i))
labelH=[0,1]


# In[6]:


def gmm(k,X,classes):
    l=[]
    R=[]
    w=np.zeros((X.shape[0],len(classes)))
    model=GaussianMixture(n_components=k,random_state=30,covariance_type='tied',max_iter=200,tol=0.1,n_init=1)
    for c in range(len(classes)):
        model.fit(classes[c])
        #z=model.score_samples(data_test[ : , :5]).reshape((-1,1))
        #R.append(z)
        w[ : ,c]+=model.score_samples(X)
    #r=np.hstack((R[i] for i in range(len(classes1)) ))
    #r=np.hstack((R[0],R[1],R[2],R[3]))
    #print(w.shape)
    for i in range(len(X)):
        z=np.argmax(w[i],axis=0)
        l.append(z)
    y_=np.array(l).reshape((-1,1)) 
    return y_   


# In[7]:


def accuracy(X,Y,classes):
    y_list=[]
    for i in [1,5,10]:
        #s=0
        #for i in range(len(Ytrain)):
         #    if Ytrain[i]==y_[i]:
          #          s+=1     
        #ac=(s/len(Ytrain))*100
        #print('acc_test=',ac)
        acc=accuracy_score(Y ,gmm(i, X,classes))
        print('accuracy of data test for k=',i,'is',acc*100)


# In[8]:


def kf(Data,label_i):
    class_list=[]
    ks=[1,5,10]
    Ys_=[]
    Y=[]
    Train_index=[]
    Test_index=[]
    ekhtelaf=[]
    s=np.zeros((75,1))
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=2652124)
    for i, j in rkf.split(Data): 
        Train_index.append(i)
        Test_index.append(j)

    for i in Train_index:
        c=[]
        w=Data[i][ : ,-1]
        for t in label_i:
            c.append(Data[i][w==t])
        class_list.append(c)    
    for i in range(25):
        for t in ks:
            Ys_.append(gmm(t,Data[Test_index[i]],class_list[i]))
            Y.append(Data[Test_index[i]][ : ,4])
    for i in range(len(Y)):
        ekhtelaf.append(Y[i]-Ys_[i])   
    for i in range(len(ekhtelaf)):
        s[i,0]+=np.count_nonzero(ekhtelaf[i])
    mean_list=[s[ :25].mean(),s[25:50].mean(),s[50: ].mean()]
    return mean_list    


# In[9]:


def bestk(mean_list):
    if min(mean_list)==mean_list[0]:
        return 1
    if min(mean_list)==mean_list[1]:
        return 5
    if min(mean_list)==mean_list[2]:
        return 10


# In[10]:


def plot_train(X0,X1,Y,title):
    colors = ['red','blue','purple','green']
    fig = plt.figure(figsize=(8,8))
    plt.scatter(X0,X1, c=Y, cmap=matplotlib.colors.ListedColormap(colors))
    plt.title(title)
    cb = plt.colorbar()
    loc = np.arange(0,max(Y),max(Y)/float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(colors)


# In[11]:


# iris data train
plot_train(train[ : ,0],train[ : ,1],train[ : ,4],'iris')


# In[12]:


# UKM data train
plot_train(data_train[ : ,0],data_train[ : ,1],data_train[ : ,5],'UKM')


# In[13]:


# vehicle data train
plot_train(datatrain[ : ,0],datatrain[ : ,1],datatrain[ : ,18],'Vehicle')


# In[14]:


# health data train
plot_train(data__train[ : ,0],data__train[ : ,1],data__train[ : ,13],'health')


# In[15]:


print('UKM')
accuracy(data_test[ : , :5],data_test[ : ,5],classes1) 
print('iris')
accuracy(test[ : , :4],test[ : ,4],classes2) 
print('Vehicle')
accuracy(datatest[ : , :19],datatest[ : ,18],classes3) 
print('Health')
accuracy(data__test[ : , :13],data__test[ : ,13],classes4) 


# In[16]:


dataUKM=np.concatenate((data_train,data_test))
kU=bestk(kf(dataUKM,labelU))
kI=bestk(kf(Data,labelI))
kV=bestk(kf(data,labelV))
kH=bestk(kf(datas,labelH))  


# In[17]:


def bestk_acc(Xtrain,Ytrain,Xtest,Ytest,classes,k):
    acctrain=accuracy_score(Ytrain ,gmm(k,Xtrain,classes))
    acctest=accuracy_score(Ytest ,gmm(kU,Xtest,classes))
    print('train accuracy for best k=',k,'is',acctrain*100)
    print('test accuracy for best k=',k,'is',acctest*100)


# In[18]:


print('UKM')
bestk_acc(data_train[ : , :5],data_train[ : ,5],data_test[ : , :5],data_test[ : ,5],classes1,kU)
print('iris')
bestk_acc(train[ : , :4],train[ : ,4],test[ : , :4],test[ : ,4],classes2,kI)
print('Vehicle')
bestk_acc(datatrain[ : , :19],datatrain[ : ,18],datatest[ : , :19],datatest[ : ,18],classes3,kV)
print('Health')
bestk_acc(data__train[ : , :13],data__train[ : ,13],data__test[ : , :13],data__test[ : ,13],classes4,kH)


# In[19]:


def plotXtest(X,classes):
    k_list=[1,5,10]
    Y_=[]
    for i in k_list:
        Y_.append(gmm(i,X,classes))
    return Y_    


# In[20]:


for i in plotXtest(data_test[ : , :5],classes1):
    plot_train(data_test[ : ,0],data_test[ : ,1],i,'UKM')  


# In[21]:


for i in plotXtest(test[ : , :4],classes2):
    plot_train(test[ : ,0],test[ : ,1],i,'iris') 


# In[22]:


for i in plotXtest(datatest[ : , :19],classes3):
    plot_train(datatest[ : ,0],datatest[ : ,1],i,'Vehicle') 


# In[23]:


for i in plotXtest(data__test[ : , :13],classes4):
    plot_train(data__test[ : ,0],data__test[ : ,1],i,'Health') 

