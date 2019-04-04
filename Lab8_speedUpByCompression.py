#from six.moves import urllib
#from sklearn.datasets import fetch_openml
#from sklearn.decomposition import PCA
#from sklearn.model_selection import train_test_split
#import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#
#
##mnist = fetch_openml('mnist_784', version=1, cache=True)
#
#
##X = mnist["data"]
##y = mnist["target"]
#
#X_train, X_test, y_train, y_test = train_test_split(X, y)
#
#
#pca = PCA(0.95) #retain 95% variance
#
#XTrain_reduced = pca.fit_transform(X_train) #compress
#XTest_reduced = pca.transform(X_test) #compress
#
##the number of components vs. the number of original features gives you an idea about compression ratio
#print(pca.n_components_) 
#
##XTrain_recovered = pca.inverse_transform(XTrain_reduced) #decompress
#
##def plot_digits(instances, images_per_row=5, **options):
##    size = 28
##    images_per_row = min(len(instances), images_per_row)
##    images = [instance.reshape(size,size) for instance in instances]
##    n_rows = (len(instances) - 1) // images_per_row + 1
##    row_images = []
##    n_empty = n_rows * images_per_row - len(instances)
##    images.append(np.zeros((size, size * n_empty)))
##    for row in range(n_rows):
##        rimages = images[row * images_per_row : (row + 1) * images_per_row]
##        row_images.append(np.concatenate(rimages, axis=1))
##    image = np.concatenate(row_images, axis=0)
##    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
##    plt.axis("off")
##    
##plt.figure(figsize=(14, 8))
##plt.subplot(121)
##plot_digits(X_train[::2100])
##plt.title("Original", fontsize=16)
##plt.subplot(122)
##plot_digits(XTrain_recovered[::2100])
##plt.title("Compressed then Decompressed", fontsize=16)
#
#from sklearn.linear_model import LogisticRegression
#import time
#
#logisticRegr = LogisticRegression(solver = 'lbfgs',max_iter = 1000, multi_class = 'multinomial')
#
#time_start = time.time()
#logisticRegr.fit(XTrain_reduced, y_train)
#print('logisticRegr done! Time elapsed: {} seconds'.format(time.time()-time_start))
#
#print('logisticRegr score: {}'.format(logisticRegr.score(XTest_reduced, y_test)))