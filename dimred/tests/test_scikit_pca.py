import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def test_np_array():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=2)
    pca.fit(X)

    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_

    print('\n[test_np_array] - Explained Variance ratio: {}'.format(explained_variance_ratio))
    print('[test_np_array] - Singular Values: {}'.format(singular_values))

    assert(explained_variance_ratio[0] == 0.9924428900898052)
    assert(explained_variance_ratio[1] == 0.007557109910194766)

    assert(singular_values[0] == 6.300612319734663)
    assert(singular_values[1] == 0.5498039617971033)

def test_mnist_data():
    # loading modified mnist dataset
    # It contains 2000 labeled images of each digit 0 and 1. Images are 28x28 pixels
    # Classes: 2 (digits 0 and 1)
    # Samples per class: 2000 samples per class
    # Samples total: 4000
    # Dimensionality: 784 (28 x 28 pixels images)
    # Features: integers calues from 0 to 255 (Pixel Grey color)
    mnist_df = pd.read_csv("data/mnist_only_0_1.csv")
    #print('MNIST Dataset sample: {}'.format(mnist_df.head()))

    pixel_colnames = mnist_df.columns[:-1]
    X = mnist_df[pixel_colnames]
    y = mnist_df['label']
    #print('X Dataset sample: {}'.format(X.head()))
    #print('y Dataset sample: {}'.format(y.head()))

    # PCA is sensitive to the scale of the features.
    # We can standardize your data onto unit scale (mean = 0 and variance = 1) by using Scikit-Learn's StandardScaler.
    scaler = StandardScaler()
    scaler.fit(X)

    pca = PCA(n_components = .90) # n_components = .90 means that scikit-learn will choose the minimum number of principal components such that 90% of the variance is retained.
    pca.fit(X)

    mnist_dimensions_before_pca = len(pixel_colnames)
    mnist_dimensions_after_pca = pca.n_components_
    print('\n[test_mnist_data] - Number of dimensions before PCA: ' + str(mnist_dimensions_before_pca))
    print('[test_mnist_data] - Number of dimensions after PCA: ' + str(mnist_dimensions_after_pca))

    assert(mnist_dimensions_before_pca == 784)
    assert(mnist_dimensions_after_pca == 48)
