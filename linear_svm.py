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

class LinearSvm():
  def __init__(self):
    mat = scipy.io.loadmat('Dataset1.mat')
    list1 = [(k, v) for k, v in mat.items()] 
    X = list1[3][1]
    Y = list1[4][1]
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size= 0.25, random_state=0)  
    #feature Scaling   
    st_x= StandardScaler()    
    self.x_train= st_x.fit_transform(self.x_train)    
    self.x_test= st_x.transform(self.x_test)    

  def classify(self, c):
    self.classifier = SVC(C=c, kernel='linear', random_state=0)  
    self.classifier.fit(self.x_train, self.y_train.ravel())  

    y_pred= self.classifier.predict(self.x_train)  
    acc = accuracy_score(self.y_train, y_pred)
    print ("accuracy train with c = ", c , " :  ", acc)

    y_pred= self.classifier.predict(self.x_test)  
    acc = accuracy_score(self.y_test, y_pred)
    print ("accuracy test with c = ", c , " :  ", acc)


  def plot(self, text, x_set, y_set):
    x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
    np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
    plt.contourf(x1, x2, self.classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
    alpha = 0.75, cmap = ListedColormap(['red', 'green']))  
    plt.xlim(x1.min(), x1.max())  
    plt.ylim(x2.min(), x2.max())  
    xLabel1 = []
    xLabel0 = []
    for i in range(len(x_set)):
      if y_set[i] == 1:
        xLabel1.append(x_set[i])
      else:
        xLabel0.append(x_set[i])

    plt.scatter(*zip(*xLabel0), color = "#cc0000", label = 0)
    plt.scatter(*zip(*xLabel1), color = "#008000", label = 1)  
    plt.title(text)  
    plt.xlabel('x0')  
    plt.ylabel('x1')  
    plt.legend()  
    plt.show()  

svmC1 = LinearSvm()
svmC1.classify(c=1)
x_set, y_set = svmC1.x_train, svmC1.y_train.ravel()  
text = 'SVM classifier (Training set) with c = 1'
svmC1.plot(text, x_set, y_set)
x_set, y_set = svmC1.x_test, svmC1.y_test.ravel()  
text = 'SVM classifier (Testing set) with c = 1'
svmC1.plot(text, x_set, y_set)

svmC100 = LinearSvm()
svmC100.classify(c=100)
x_set, y_set = svmC100.x_train, svmC100.y_train.ravel()  
text = 'SVM classifier (Training set) with c = 100'
svmC100.plot(text, x_set, y_set)
x_set, y_set = svmC100.x_test, svmC100.y_test.ravel()  
text = 'SVM classifier (Testing set) with c = 100'
svmC100.plot(text, x_set, y_set)