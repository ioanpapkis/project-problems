import matplotlib.pyplot as plt 
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import cross_val_score

import matplotlib.image as mpimg
import matplotlib.pyplot as plt



lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)#min_faces_per_person=70, resize=0.4)


n_samples, h, w = lfw_people.images.shape


X = lfw_people.data
n_features = X.shape[1]

people_images = lfw_people.images

y = lfw_people.target

target_names = lfw_people.target_names
n_classes = target_names.shape[0]

#train test split for lfw
X_train1, X_test1, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

##### Standardize the data #######
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train1)
X_test = scaler.fit_transform(X_test1)

#### number of eigenfaces to keep####
n_comp = list(range(10,210,10))

n_components =120

######## PCA #############
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True,random_state=42).fit(X_train)
eigenfaces = pca.components_.reshape((n_components, h, w))
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)



######### LDA and SVM ############
lda = LinearDiscriminantAnalysis()
lda.fit(X_train,y_train) # lda model training
X_train_lda = lda.transform(X_train) # Conversion 
X_test_lda = lda.transform(X_test) # Conversion


param_grid = {'C': [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=10)
clf = clf.fit(X_train_lda, y_train)
Best_estimator = clf.best_estimator_
Best_score = clf.best_score_
y_pred = clf.predict(X_test_lda)
report = classification_report(y_pred,y_test,target_names=target_names)


print('lda+svm accuracy:', Best_score)

#### plot accuracy test for lda svm####
n_c = list(range(1,7,1))
lda_L = []
for i in n_c:
    lda2= LinearDiscriminantAnalysis(n_components = i)
    lda2.fit(X_train,y_train)
    X_train_lda2 =  lda2.transform(X_train)
    X_test_lda2 = lda2.transform(X_test)
    svm = SVC(C=100.0, cache_size=200, class_weight='balanced', coef0=0.0,decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    lda_svm = svm.fit(X_train_lda2,y_train)
    y_pred = lda_svm.predict(X_test_lda2)
    lda_SVM_score = lda_svm.score(X_test_lda2,y_test)
    lda_L.append(lda_SVM_score)
    
plt.plot(n_c,lda_L)
plt.show()


##### PCA + SVM with best estimator ######

# parameter grid for grid search 
#param_grid={'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
#      'C':[1e3, 1e2, 1e4, 5e4, 1e5],
#      'degree':np.arange(3,6),   
#      'coef0':np.arange(0.001,3,0.5),
#      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1] }

#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
# best estimator for svm model ####
svm = SVC(C=1e4,kernel = 'poly',degree = 4,coef0=1.501,gamma = 0.005)
pca_svm = svm.fit(X_train_pca,y_train)
y_pred = pca_svm.predict(X_test_pca)
PCA_SVM_score = pca_svm.score(X_test_pca,y_test)
report = classification_report(y_test,y_pred,target_names=target_names)

print('PCA + svm :',PCA_SVM_score)
print(report)


#### plot pca and svm accuracy test ####
svm_L = []
for i in n_comp:
    pca = PCA(n_components=i, svd_solver='randomized',whiten=True,random_state=42).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    svm = SVC(C=1e4,kernel = 'poly',degree = 4,coef0=1.501,gamma = 0.005)
    pca_svm = svm.fit(X_train_pca,y_train)
    y_pred = pca_svm.predict(X_test_pca)
    PCA_SVM_score = pca_svm.score(X_test_pca,y_test)
    svm_L.append(PCA_SVM_score)
    
plt.plot(n_comp,svm_L)
plt.show()
###################### KNN ###################################
#find the optimal number k with 10-fold cv 

# PCA and Knn 
neighbors = list(range(1,50,2))
# empty list that will hold cv scores
cv_scores_pca = [ ]
#perform 10-fold cross-validation
for K in neighbors:
    knn = KNeighborsClassifier(n_neighbors = K)
    scores = cross_val_score(knn,X_train_pca,y_train,cv = 10,scoring =
    "accuracy")
    cv_scores_pca.append(scores.mean())


# Changing to mis classification error
mse = [1-x for x in cv_scores_pca]
# determing best k
optimal_k_pca = neighbors[mse.index(min(mse))]
print("The optimal no. of neighbors(PCA) is {}".format(optimal_k_pca))


plt.scatter(neighbors,cv_scores_pca)
plt.xlabel('Accuracy')
plt.ylabel('k')
plt.title('PCA and Knn')
plt.show()


# PCA and Knn 
neighbors = list(range(1,50,2))
# empty list that will hold cv scores
cv_scores_lda = [ ]
#perform 10-fold cross-validation
for K in neighbors:
    knn = KNeighborsClassifier(n_neighbors = K)
    scores = cross_val_score(knn,X_train_lda,y_train,cv = 10,scoring =
    "accuracy")
    cv_scores_lda.append(scores.mean())


# Changing to mis classification error
mse = [1-x for x in cv_scores_lda]
# determing best k
optimal_k_lda = neighbors[mse.index(min(mse))]
print("The optimal no. of neighbors(LDA) is {}".format(optimal_k_lda))


plt.scatter(neighbors,cv_scores_lda)
plt.xlabel('Accuracy')
plt.ylabel('k')
plt.title('LDA and Knn')
plt.show()



# lda + knn
knn = KNeighborsClassifier(n_neighbors=optimal_k_lda)  
knn.fit(X_train_lda, y_train) 
print("Dimension reduction using LDA, Test set accuracy: {: .2f}".format(knn.score(X_test_lda, y_test)))


# pca + knn
knn2 = KNeighborsClassifier(n_neighbors=optimal_k_pca)
knn2.fit(X_train_pca,y_train)
print('pca+knn:',knn2.score(X_test_pca,y_test))



