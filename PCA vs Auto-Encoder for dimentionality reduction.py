#PCA vs AE wich one is better to generate data 
#import librairies
import matplotlib 
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'
#
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 

from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error,silhouette_score
import matplotlib._color_data as mcd


'''
A challenging task in the modern 'Big Data' era is to reduce the feature space since it is very computationally expensive to perform any kind of analysis or modelling in today's extremely big data sets. There is variety of techniques out there for this purpose: PCA, LDA, Laplacian Eigenmaps, Diffusion Maps, etc...Here I make use of a Neural Network based approach, the Autoencoders. An autoencoder is essentially a Neural Network that replicates the input layer in its output, after coding it (somehow) in-between. In other words, the NN tries to predict its input after passing it through a stack of layers. The actual architecture of the NN is not standard but is user-defined and selected. Usually it seems like a mirrored image (e.g. 1st layer 256 nodes, 2nd layer 64 nodes, 3rd layer again 256 nodes).'''
#data creation and scatter
X , y= make_blobs(n_features=50,centers=20,n_samples=20000,cluster_std=0.2,center_box=[-1,1],random_state=17)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=17)
scaler = MinMaxScaler()
X_train =scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

#Reduce data dimention from 50 to 2 dimention 
pca = PCA(n_components=2)
pca.fit(X_train)
res_pca = pca.transform(X_test)
# res_pca.shape : (10% of data = 2000, 2 dim from 50)

from keras.models import Model


unique_labels = np.unique(y_test)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
cmap = get_cmap((20))
print(cmap(1))
for index,unique_label in enumerate(unique_labels):
    X_data = res_pca[y_test==unique_label]
    plt.scatter(X_data[:,0],X_data[:,1],alpha=0.3,c =cmap(index))

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA results")


####### AUTO ENCODER IMPLEMENTATION

autoencoder= MLPRegressor(alpha=1e-15,
                          hidden_layer_sizes=(50,100,50,2,50,100,50),
                          random_state=1,
                          max_iter=200)

autoencoder.fit(X_train,X_train)


W = autoencoder.coefs_
biases= autoencoder.intercepts_
for i in W :
    print(i)
W = W[0:4]
biases= biases[0:4]

    
    
def encode(encoder_weights,encoder_biases,data):
    res=data
    for index, (w,b) in enumerate(zip(encoder_weights,encoder_biases)):
        if index+1 == len(encoder_weights):
            res= np.dot(res,w) + b
        else:
            res  = np.maximum(0,np.dot(res,w)  + b)
    return res

res = encode(W,biases,X_test)
print(res.shape)

unique_labels = np.unique(y_test)
for index,unique_label in enumerate(unique_labels):
    data_latent_space = res[y_test==unique_label]
    plt.scatter(data_latent_space[:,0],data_latent_space[:,1],alpha=0.3,c =cmap(index))

plt.xlabel("Latent X")
plt.ylabel("Latest Y")
plt.title("Autoencoder results")

print(silhouette_score(res,y_test))
print("PCA silouette score( how good the clustering is made")
print(silhouette_score(res_pca,y_test))
