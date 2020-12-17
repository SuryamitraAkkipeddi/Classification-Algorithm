###########################################################################################
#################################### I> INSTALL PACKAGES ##################################
###########################################################################################

install.packages('caTools') 
install.packages('ggplot2')
install.packages('e1071')
install.packages('rpart')
install.packages('randomForest')
install.packages('ElemStatLearn')
install.packages('class')
install.packages('caret')
install.packages('kernlab')
install.packages('MASS')

###########################################################################################
#################################### II> LOAD THE LIBRARIES ###############################
###########################################################################################

library(caTools)
library(ggplot2)
library(e1071)
library(rpart)
library(randomForest)
library(ElemStatLearn)
library(class)
library(caret)
library(kernlab)
library(MASS)
set.seed(123)

###########################################################################################
#################################### III> IMPORT THE DATASET  #############################
###########################################################################################

dataset = read.csv('E:\\MACHINE LEARNING SUMMARY\\ML DATASETS\\Social_Network_Ads.csv')  
summary(dataset)

###########################################################################################
################################### V> OMIT UNNECESSARY DATA ##############################
###########################################################################################

dataset = dataset[3:5]

###########################################################################################
################################### VI> ENCODING THE CATEGORICAL DATA #####################
###########################################################################################

dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

###########################################################################################
################################### VII> SPLIT THE DATASET INTO TRAIN SET AND TEST SET ####
###########################################################################################

split = sample.split(dataset$Purchased, SplitRatio = 0.75)                                                 
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

###########################################################################################
################################## VIII> FEATURE SCALING ##################################
###########################################################################################

training_set[-3] = scale(training_set[-3])                                                          
test_set[-3]     = scale(test_set[-3])

###########################################################################################
################################## XI> FIT/TRAIN ML MODEL TO THE TRAINING SET #############
###########################################################################################

classifier = rpart(formula = Purchased ~ .,data = training_set)                                                     # decision tree classification

classifier = svm(formula = Purchased ~ .,data = training_set,type = 'C-classification',kernel = 'radial')           # kernel SVM

y_pred = knn(train = training_set[, -3],test = test_set[, -3],cl = training_set[, 3],k = 5,prob = TRUE)             # k-NN (FITS AND PREDICTS DIRECTLY-THIS STEP)

classifier = glm(formula = Purchased ~ ., family = binomial, data = training_set)                                   # logistic regression

classifier = naiveBayes(x = training_set[-3], y = training_set$Purchased)                                           # naive bayes

classifier = randomForest(x = training_set[-3], y = training_set$Purchased, ntree = 500)                            # random forest

classifier = svm(formula = Purchased ~ .,data = training_set,type = 'C-classification',kernel = 'linear')           # SVM

###########################################################################################
############################XII> PREDICT THE TEST SET RESULTS #############################
###########################################################################################

y_pred = predict(classifier, newdata = test_set[-3], type = 'class')                                          # decision tree classification

prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])                                    # logistic regression
y_pred = ifelse(prob_pred > 0.5, 1, 0)

y_pred = predict(classifier, newdata = test_set[-3])                                                          # random forest, kernel SVM, SVM, naive bayes 

###########################################################################################
############################ XIII> CONFUSION MATRIX #######################################
###########################################################################################

cm = table(test_set[, 3], y_pred)                 # y_pred >0.5 for logistic regression and y_pred for rest all classifiers                                              

###########################################################################################
################################### XV> VISUALIZE TRAIN SET  ##############################
###########################################################################################

set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')

prob_set = predict(classifier, type = 'response', newdata = grid_set)                    # logistic regression
y_grid = ifelse(prob_set > 0.5, 1, 0)                                                    # logistic regression

y_grid = knn(train = training_set[, -3], test = grid_set, cl = training_set[, 3], k = 5) # k-NN

y_grid = predict(classifier, newdata = grid_set)                                         # SVM, kernel SVM, Naive Bayes, Random Forest

y_grid = predict(classifier, newdata = grid_set, type = 'class')                         # Decision Tree

plot(set[, -3],
     main = 'Training set',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

###########################################################################################
################################### XVI> VISUALIZE TEST SET  ##############################
###########################################################################################

set = test_set                                                                                              
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')

y_grid = predict(classifier, newdata = grid_set, type = 'class')                          # decision tree classification

y_grid = predict(classifier, newdata = grid_set)                                          # kernel SVM, SVM, Naive bayes

y_grid = knn(train = training_set[, -3], test = grid_set, cl = training_set[, 3], k = 5)  # k-Nearest Neighbors

prob_set = predict(classifier, type = 'response', newdata = grid_set)                     # logistic regression
y_grid = ifelse(prob_set > 0.5, 1, 0)                                                     # logistic regression

y_grid = predict(classifier, grid_set)                                                    # random forest

plot(set[, -3], main = 'Classification model (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

###########################################################################################
################################### XVII> PLOTTING  #######################################
###########################################################################################

plot(classifier)                                  # decision tree, random forest
text(classifier)                                  # decision tree, random forest


