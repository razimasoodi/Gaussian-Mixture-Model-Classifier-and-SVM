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
from sklearn.model_selection import cross_val_score

class KernelSvm():
  def __init__(self):
    datContent = [i.strip().split() for i in open("Health.dat").readlines()]
    data = np.array(datContent)
    X = data[:,0:13]
    Y = data[:,13]
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size= 0.25, random_state=0)  
    #feature Scaling   
    st_x= StandardScaler()    
    self.x_train= st_x.fit_transform(self.x_train)    
    self.x_test= st_x.transform(self.x_test)    

  def mKFoldValidation(self, c, std, k, isTrain):
    self.classifier = SVC(C=c, kernel='rbf', gamma=(1/std), random_state=0)  
    if isTrain:
      scores = cross_val_score(self.classifier, self.x_train, self.y_train, cv=k)
    else:
      scores = cross_val_score(self.classifier, self.x_test, self.y_test, cv=k)
    return (scores.mean(), scores.std())

  def mKTimeKFold(self, c, std, k, isTrain):
    accs = 0
    stdAccs = 0
    for _ in range(k):
      m , s = self.mKFoldValidation(c, std, k, isTrain)
      accs += m
      stdAccs += s
    return (accs/k, stdAccs/k)

  def find_best_params(self, k):
    values = [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]
    accuracies = []
    for item in values:
      for item2 in values:
        meanAccs, _ = self.mKTimeKFold(item, item2, k, True)
        accuracies.append((item, item2, meanAccs))
    accuracies.sort(key=lambda tup: tup[2], reverse=True)
    print ("best c is ",accuracies[0][0],"and best std is",accuracies[0][1],"with",accuracies[0][2],"accuracy")
    return(accuracies[0][0], accuracies[0][1])

  def classify(self, c, std):
    self.classifier = SVC(C=c, kernel='rbf', gamma=(1/std), random_state=0)  
    self.classifier.fit(self.x_train, self.y_train)  

    y_pred= self.classifier.predict(self.x_train)  
    cm = confusion_matrix(self.y_train, y_pred) 
    accuracy = (cm[0][0] + cm[1][1])*100 / (cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
    print ("accuracy train with c = ", c ," and std = ", std , " :  ", accuracy)

    y_pred= self.classifier.predict(self.x_test)  
    cm = confusion_matrix(self.y_test, y_pred) 
    accuracy = (cm[0][0] + cm[1][1])*100 / (cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
    print ("accuracy test with c = ", c ," and std = ", std , " :  ", accuracy)


  def plotAccuracy(self):
    values = [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]
    accuraciesTrain = []
    stdAccuraciesTrain = []
    accuraciesTest = []
    stdAccuraciesTest = []
    for c in values:
      for std in values:
        meanAccs, stdAccs = self.mKTimeKFold(c, std, 5, True)
        accuraciesTrain.append([c, std, meanAccs])
        stdAccuraciesTrain.append([c, std, stdAccs])
        meanAccs, stdAccs = self.mKTimeKFold(c, std, 5, False)
        accuraciesTest.append([c, std, meanAccs])
        stdAccuraciesTest.append([c, std, stdAccs])

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
    

svmC1 = KernelSvm()
svmC1.plotAccuracy()
svmC1 = KernelSvm()
c, std = svmC1.find_best_params(k=10)
svmC1.classify(c, std)
