# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:21:05 2021

@author: johnp
"""
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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


img = mpimg.imread('κουλης1.png')

    
gray = rgb2gray(img)

    
plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()






def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)



def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())








#people = fetch_lfw_people(min_faces_per_person=70, resize = 1) # Read the face data of the data set sklearn (200MB size)

lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.5)#min_faces_per_person=70, resize=0.4)


n_samples, h, w = lfw_people.images.shape


X = lfw_people.data
n_features = X.shape[1]



#X1 = people.images

s = lfw_people.images
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

#train test split for lfw
X_train1, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train1)

n_components = 110

lda = LinearDiscriminantAnalysis()
lda.fit(X_train,y_train) # lda model training
X_train_lda = lda.transform(X_train) # Conversion 
X_test_lda = lda.transform(X_test) # Conversion



param_grid = {'C': [1e1, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],'kernel':['rbf', 'poly', 'sigmoid'] }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=10)
clf = clf.fit(X_train_lda, y_train)
#Best_estimator = clf.best_estimator_
Best_score = clf.best_score_
y_pred = clf.predict(X_test_lda)
#report = classification_report(y_test, y_pred, target_names=target_names)
#matrix = confusion_matrix(y_test, y_pred, labels=range(n_classes))


print('lda+svm accuracy:', Best_score)





pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((n_components, h, w))
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


clf2 = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=10)
clf2 = clf2.fit(X_train_pca, y_train)
Best_estimator = clf2.best_estimator_ 
pca_svm = clf2.best_score_
print(Best_estimator)
y_pred_pca = clf2.predict(X_test_pca)
report = classification_report(y_test, y_pred_pca, target_names=target_names)
print(report)

print('PCA+SVM accuracy:',pca_svm)


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
print("The optimal no. of neighbors is {}".format(optimal_k_pca))


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
print("The optimal no. of neighbors is {}".format(optimal_k_lda))


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









