
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA





lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)#min_faces_per_person=70, resize=0.4)


n_samples, h, w = lfw_people.images.shape


X = lfw_people.data
n_features = X.shape[1]


s = lfw_people.images
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

#train test split for lfw
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

n_components = 100

pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
components = pca.transform(X_train)
projected = pca.inverse_transform(components)  



def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


### data visualisation after dim reduction#####
# resize 0.4 ----> scale to 50x37 ####### resize 0.5 -------> scale to 62x47 
fig, ax = plt.subplots(2, 2, figsize=(2, 6),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(2):
    ax[0, i].imshow(X_train[i].reshape(50,37), cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(50,37), cmap='binary_r')
    
ax[0, 0].set_ylabel('full-dim\ninput')
ax[1, 0].set_ylabel('image after dim\nreduction');
plt.show()






##### eigenfaces ###### 
eigenfaces = pca.components_.reshape((n_components, h, w))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()

##### explained variance of eigenvalues ######
pca2 = PCA(n_components=n_components).fit(X_train)
plt.plot(np.cumsum(pca2.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title('Principal Component Analysis(PCA)')
plt.show()








