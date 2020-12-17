# classification: logistic regression, decision tree, kernel SVM(support vector machine), K-NN, Naive Bayes, SVM, Random forest

###################################################################################################################################
#######################  I> IMPORT THE LIBRARIES  ################################################################################# 
###################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


###################################################################################################################################
#######################  II> IMPORT THE DATASET & DATA PREPROCESSING   ############################################################ 
###################################################################################################################################

dataset = pd.read_csv('E:\\DESK PROJECTS\\MACHINE LEARNING SUMMARY\\ML DATASETS\\Social_Network_Ads.csv')   
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


###################################################################################################################################
###################### III> SPLIT DATASET INTO TRAIN AND TEST SETS  ###############################################################
###################################################################################################################################

from sklearn.model_selection import train_test_split                                          
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


###################################################################################################################################
##################### IV>  FEATURE SCALING     ####################################################################################
###################################################################################################################################

from sklearn.preprocessing import StandardScaler                                                                 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


###################################################################################################################################
####################### V> FIT ML MODEL TO TRAINING SET ###########################################################################
###################################################################################################################################

from sklearn.linear_model import LogisticRegression                                           # logistic regression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier                                               # decision tree classification
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.svm import SVC                                                                   # kernel SVM
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier                                            # k-Nearest Neighbors 
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

from sklearn.naive_bayes import GaussianNB                                                    # Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, y_train)

from sklearn.svm import SVC                                                                   # Support Vector Machine
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier                                           # random forest
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


##################################################################################################################################
####################### VI> PREDICT TEST SET RESULTS #############################################################################
##################################################################################################################################
    
y_pred = classifier.predict(X_test)             

                                            
##################################################################################################################################
#######################  VII> CONFUSION MATRIX      ##############################################################################
##################################################################################################################################

from sklearn.metrics import confusion_matrix                                                   
cm = confusion_matrix(y_test, y_pred)
cm


##################################################################################################################################
#######################  VIII> VISUALIZE TRAINING SET RESULTS ####################################################################
##################################################################################################################################

from matplotlib.colors import ListedColormap 
X_set, y_set = X_train, y_train                                                  
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classification model (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


##################################################################################################################################
#######################  IX> VISUALIZE TEST SET RESULTS ##########################################################################
##################################################################################################################################

from matplotlib.colors import ListedColormap                                                        
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classification model (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
