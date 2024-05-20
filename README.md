# Gaussian-Mixture-Model-Classifier-and-SVM
Part A: Gaussian Mixture Model Classifier

Datasets: User Knowledge Modeling Data Set (UKM.xls), Iris, Vehicle.dat, Health.dat

In this part, Gaussian Mixture Model (GMM) is used as a generative classifier. You can use the GMM toolbox in MATLAB or scikit-learn library from python which uses the Expectation Maximization (EM) to train a GMM model. A GMM model can be employed to estimate the PDF of some samples (like a parametric density estimator). Here, you should train an individual GMM model (with K Components) for each class. Therefore, N GMM models will be created where N shows the number of classes. The label of a sample can be determined using Maximum Likelihood (ML) criteria. In another words, you should find the likelihood of a sample in all classes and then select the class with the maximum likelihood as the label of the sample.

• Plot the training data (Different colors for each class).

• Construct a GMM classifier, with K = 1,5,10, Gaussian components and train on Train Data.

• For each k, plot the test data classified by the GMM classifier.

• Use five-time-five-fold cross validation to determine the best K.





Part B: Support Vector Machine (SVM) 

Linear SVM:

Use “Dataset1.mat” which is a 2D and 2-class dataset to do this part. The dataset has been attached to the project.

• Train the SVM using two different values of the penalty parameter, i.e., C=1 and C=100.

• Plot the data and the decision boundary.

• Report the train accuracy for both C=1 and C=100.

Kernel SVM for two-class problem:

In general, SVM is a linear classifier. When data are not linearly separable, Kernel SVM can be used. Here, you will utilize SVM with RBF kernel for non-linear classification. Perform the following step for “Dataset2.mat” and “Health.dat” datasets.

• Train SVM with the penalty parameter C and the standard deviation for RBF kernel. Determine the best value C by ten-time-ten-fold cross validation. Note: It is better to test the values in multiplicative steps such as 0.01, 0.04, 0.1, 0.4, 1, 4, 10 and 40. Therefore, you should evaluate 64 (82) different models to select the best model.

• Plot train and test accuracies and their corresponding variances of five-time-five-fold cross validation for different values of C and 𝜎.

• Plot the data and the decision boundary for “Dataset2.mat” (for best model)

• Report the test accuracy using the selected model (best C and 𝜎 )

Kernel SVM for multi-class problem:

As you know, SVM is a two-class classifier. However, it can be extended for multi-class
classification. For this, two approaches are possible: one-against-one and one-against-all
(one-against-rest). Use one-against-all method (Dataset:Vehicle.dat)

• Design a multi-class SVM classifier with one-against-all method.

• Determine the best value of C by cross validation

• Plot the train and test accuracies and their corresponding variances of five-time-five-fold cross validation for different value of C and 𝜎.

• Report your test accuracy using the selected model. (best C and 𝜎)

