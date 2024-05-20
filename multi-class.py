import csv
import scipy.io
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from random import randint
from sklearn.multiclass import OneVsRestClassifier
import statistics
import sys
np.set_printoptions(threshold=sys.maxsize)

class MultiKernelSvm():
  def __init__(self):
    datContent = [i.strip().split() for i in open("Vehicle.dat").readlines()]
    data = np.array(datContent)
    # print((data.shape)) (94, 19)
    X = data[:,:18]
    Y = data[:,18]
    van = 0
    sab = 0
    opel = 0
    bus = 0
    for i in range(len(Y)):
      if Y[i] == 'van' :
        Y[i] = 1
        van += 1
      elif Y[i] == 'saab':
        Y[i] = 2
        sab += 1
      elif Y[i] == 'opel':
        Y[i] = 3
        opel += 1
      else:
        Y[i] = 4
        bus += 1
    #print (van, sab, opel, bus) 28 20 20 26
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)     
    st_x= StandardScaler()    
    self.x_train= st_x.fit_transform(self.x_train)    
    self.x_test= st_x.transform(self.x_test) 
    # print(self.x_test.shape) (19, 18)

  def classify2(self, c, std, x_train, y_train, x_test, y_test):
    self.classifier = SVC(C=c, kernel='rbf', gamma=(1/std), random_state=0)  
    ovr = OneVsRestClassifier(self.classifier)
    ovr.fit(x_train, y_train)  
    y_pred= ovr.predict(x_test)  
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

  def classify(self, c, std, x_train, y_train, x_test, y_test):
    # van vs all
    y1All = []
    for i in range(len(y_train)):
      if y_train[i] == '1' :
        y1All.append(1)
      else:
        y1All.append(2)
    # print(y1All)
    classifier1VsAll = SVC(C=c, kernel='rbf', gamma=(1/std), random_state=0)  
    classifier1VsAll.fit(x_train[:], y1All)  
    y_pred1VsAll= classifier1VsAll.predict(x_test[:])  

    # saab vs all
    y2All = []
    for i in range(len(y_train)):
      if y_train[i] == '2' :
        y2All.append(1)
      else:
        y2All.append(2)
    classifier2VsAll = SVC(C=c, kernel='rbf', gamma=(1/std), random_state=0)  
    classifier2VsAll.fit(x_train[:], y2All)  
    y_pred2VsAll= classifier2VsAll.predict(x_test[:]) 

    # opel vs all
    y3All = []
    for i in range(len(y_train)):
      if y_train[i] == '3' :
        y3All.append(1)
      else:
        y3All.append(2)
    classifier3VsAll = SVC(C=c, kernel='rbf', gamma=(1/std), random_state=0)  
    classifier3VsAll.fit(x_train[:], y3All)  
    y_pred3VsAll= classifier3VsAll.predict(x_test[:]) 

    # bus vs all
    y4All = []
    for i in range(len(y_train)):
      if y_train[i] == '4' :
        y4All.append(1)
      else:
        y4All.append(2)
    classifier4VsAll = SVC(C=c, kernel='rbf', gamma=(1/std), random_state=0)  
    classifier4VsAll.fit(x_train[:], y4All)  
    y_pred4VsAll= classifier4VsAll.predict(x_test[:]) 

    y_pred = []
    for i in range(len(y_pred1VsAll)):
      pred = [y_pred1VsAll[i], y_pred2VsAll[i], y_pred3VsAll[i], y_pred4VsAll[i]]
      # print("t",y_test[i] ,pred)
      # ind = randint(0,3)
      ind = np.random.choice(np.arange(0, 4), p=[28/94, 20/94, 20/94, 26/94])
      if 1 in pred:
        ind = pred.index(1)
      y_pred.append(ind+1)

    y_test = [int(i) for i in y_test] 
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

  def m10Fold(self, c, s):
    x = np.concatenate((self.x_train, self.x_test), axis=0)
    y = np.concatenate((self.y_train, self.y_test), axis=0)
    x_test = x[:9]
    y_test = y[:9]
    x_train = x[9:]
    y_train = y[9:]
    accuracy10 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy1 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[9:18]
    y_test = y[9:18]
    x_train = np.concatenate((x[18:], x[:9]), axis=0)
    y_train = np.concatenate((y[18:], y[:9]), axis=0)
    accuracy20 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy2 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[18:27]
    y_test = y[18:27]
    x_train = np.concatenate((x[27:], x[:18]), axis=0)
    y_train = np.concatenate((y[27:], y[:18]), axis=0)
    accuracy30 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy3 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[27:37]
    y_test = y[27:37]
    x_train = np.concatenate((x[37:], x[:27]), axis=0)
    y_train = np.concatenate((y[37:], y[:27]), axis=0)
    accuracy40 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy4 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[37:46]
    y_test = y[37:46]
    x_train = np.concatenate((x[46:], x[:37]), axis=0)
    y_train = np.concatenate((y[46:], y[:37]), axis=0)
    accuracy50 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy5 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[46:55]
    y_test = y[46:55]
    x_train = np.concatenate((x[55:], x[:46]), axis=0)
    y_train = np.concatenate((y[55:], y[:46]), axis=0)
    accuracy60 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy6 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[55:65]
    y_test = y[55:65]
    x_train = np.concatenate((x[65:], x[:55]), axis=0)
    y_train = np.concatenate((y[65:], y[:55]), axis=0)
    accuracy70 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy7 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[65:74]
    y_test = y[65:74]
    x_train = np.concatenate((x[74:], x[:65]), axis=0)
    y_train = np.concatenate((y[74:], y[:65]), axis=0)
    accuracy80 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy8 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[74:83]
    y_test = y[74:83]
    x_train = np.concatenate((x[83:], x[:74]), axis=0)
    y_train = np.concatenate((y[83:], y[:74]), axis=0)
    accuracy90 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy9 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[83:]
    y_test = y[83:]
    x_train = x[:83]
    y_train = y[:83]
    accuracyA0 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracyA = self.classify(c, s, x_train, y_train, x_test, y_test)
    accs = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracyA, accuracy10, accuracy20, accuracy30, accuracy40, accuracy50, accuracy60, accuracy70, accuracy80, accuracy90, accuracyA0]
    return (sum(accs)/len(accs), statistics.stdev(accs))


  def m5Fold(self, c, s):
    x = np.concatenate((self.x_train, self.x_test), axis=0)
    y = np.concatenate((self.y_train, self.y_test), axis=0)
    x_test = x[:18]
    y_test = y[:18]
    x_train = x[18:]
    y_train = y[18:]
    accuracy10 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy1 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[18:36]
    y_test = y[18:36]
    x_train = np.concatenate((x[36:], x[:18]), axis=0)
    y_train = np.concatenate((y[36:], y[:18]), axis=0)
    accuracy20 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy2 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[36:55]
    y_test = y[36:55]
    x_train = np.concatenate((x[55:], x[:36]), axis=0)
    y_train = np.concatenate((y[55:], y[:36]), axis=0)
    accuracy30 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy3 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[55:74]
    y_test = y[55:74]
    x_train = np.concatenate((x[74:], x[:55]), axis=0)
    y_train = np.concatenate((y[74:], y[:55]), axis=0)
    accuracy40 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy4 = self.classify(c, s, x_train, y_train, x_test, y_test)

    x_test = x[74:]
    y_test = y[74:]
    x_train = x[:74]
    y_train = y[:74]
    accuracy50 = self.classify(c, s, x_train, y_train, x_train, y_train)
    accuracy5 = self.classify(c, s, x_train, y_train, x_test, y_test)
    accsTest = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5]
    accsTrain = [accuracy10, accuracy20, accuracy30, accuracy40, accuracy50]
    return (sum(accsTrain)/len(accsTrain), statistics.stdev(accsTrain), sum(accsTest)/len(accsTest), statistics.stdev(accsTest))


  def find_best_params(self, k):
    values = [0.001, 0.01, 0.04, 0.1, 0.4, 1, 4, 20, 10, 40, 100]
    accuracies = []
    for c in values:
      for s in values:
        meanAccs = 0
        for _ in range(k):
          m, _ = self.m10Fold(c, s)
          meanAccs += m
        accuracies.append((c, s, meanAccs/k))
    accuracies.sort(key=lambda tup: tup[2], reverse=True)
    print ("best c is ",accuracies[0][0],"and best std is",accuracies[0][1],"with",accuracies[0][2],"accuracy")
    return(accuracies[0][0], accuracies[0][1])

  def plotAccuracy(self):
    values = [0.001, 0.01, 0.04, 0.1, 0.4, 1, 4, 20, 10, 40, 100]
    accuraciesTrain = []
    stdAccuraciesTrain = []
    accuraciesTest = []
    stdAccuraciesTest = []
    for c in values:
      for std in values:
        meanAccsTrain, stdAccsTrain, meanAccsTest, stdAccsTest, = self.m5Fold(c, std)
        accuraciesTrain.append([c, std, meanAccsTrain])
        stdAccuraciesTrain.append([c, std, stdAccsTrain])
        accuraciesTest.append([c, std, meanAccsTest])
        stdAccuraciesTest.append([c, std, stdAccsTest])

    accuraciesTrain = np.array(accuraciesTrain)
    stdAccuraciesTrain = np.array(stdAccuraciesTrain)
    accuraciesTest = np.array(accuraciesTest)
    stdAccuraciesTest = np.array(stdAccuraciesTest)

    ax = plt.axes(projection ="3d")
    ax.scatter3D(accuraciesTrain[:,0], accuraciesTrain[:,1], accuraciesTrain[:,2], color = "green", label= "train")
    ax.scatter3D(accuraciesTest[:,0], accuraciesTest[:,1], accuraciesTest[:,2], color = "blue", label="test")
    plt.title("train and test accuracies for different values of C and 𝜎.")
    ax.set_xlabel('C') 
    ax.set_ylabel('std') 
    ax.set_zlabel('accuracy')
    ax.legend()
    plt.show()

    ax = plt.axes(projection ="3d")
    ax.scatter3D(stdAccuraciesTrain[:,0], stdAccuraciesTrain[:,1], stdAccuraciesTrain[:,2], color = "green", label= "train")
    ax.scatter3D(stdAccuraciesTest[:,0], stdAccuraciesTest[:,1], stdAccuraciesTest[:,2], color = "blue", label="test")
    plt.title("train and test variances for different values of C and 𝜎.")
    ax.set_xlabel('C') 
    ax.set_ylabel('std') 
    ax.set_zlabel('variance')
    ax.legend()
    plt.show()
 

m = MultiKernelSvm()
m.plotAccuracy()

m = MultiKernelSvm()
cBest, stdBest = m.find_best_params(k=10)
print("accuracy train", m.classify(cBest, stdBest, m.x_train, m.y_train, m.x_train, m.y_train))
print("accuracy test", m.classify(cBest, stdBest, m.x_train, m.y_train, m.x_test, m.y_test))
