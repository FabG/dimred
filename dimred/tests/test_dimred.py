import os
import numpy as np
from numpy import count_nonzero
import pandas as pd
from dimred import DimRed
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, isspmatrix

# Set up absolute path to unit test files
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
#MY_DATA_PATH = os.path.join(THIS_DIR, os.pardir, 'data/data.csv')
MY_DATA_PATH_MNIST = os.path.join(THIS_DIR, 'data/mnist_only_0_1.csv')
MY_DATA_PATH_IRIS = os.path.join(THIS_DIR, 'data/iris_data.csv')
print('MY_DATA_PATH_MNIST = {}'.format(MY_DATA_PATH_MNIST))


def test_np_array():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    dimred = DimRed(n_components=2)
    model = dimred.fit(X)
    explained_variance_ratio = dimred.explained_variance_ratio_

    print('\n[test_np_array] - Explained Variance ratio: {}'.format(explained_variance_ratio))

    assert(explained_variance_ratio[0] == 0.9924428900898052)
    assert(explained_variance_ratio[1] == 0.007557109910194766)

def test_np_array_sparse_noncsr():
    # create sparse matrix
    X_sparse = np.array([[1,0,0,0,0,0], [0,0,2,0,0,0], [0,0,0,2,0,0]])

    # calculate sparsity
    sparsity = 1.0 - count_nonzero(X_sparse) / X_sparse.size
    # The above array has 0.833 sparsity (meaning 83.3% of its values are 0)
    print('\n[test_np_array_sparse_noncsr] - Checking no exception for sparsity of: {}'.format(sparsity))

    dimred = DimRed(n_components=1)
    try:
        dimred.fit(X_sparse)
        assert True
        explained_variance_ratio = dimred.explained_variance_ratio_
        singular_values = dimred.singular_values_

        print('\n[test_np_array] - Explained Variance ratio: {}'.format(explained_variance_ratio))
        print('[test_np_array] - Singular Values: {}'.format(singular_values))

        assert(explained_variance_ratio[0] == 0.6666666666666667)
        assert(singular_values[0] == 2.0000000000000004)
    except TypeError:
        assert False

def test_np_array_sparse_csr():
    # create sparse matrix
    X_sparse = csr_matrix((3, 4))
    X_sparse_array = csr_matrix((3, 4), dtype=np.int8).toarray()
    # calculate sparsity
    sparsity = 1.0 - count_nonzero(X_sparse_array) / X_sparse_array.size
    # The above array has 1.0 sparsity (meaning 100% of its values are 0)
    print('\n[test_np_array_sparse_csr] - Checking compressed sparse exception for sparsity of: {}'.format(sparsity))

    dimred = DimRed(n_components=2)
    try:
        dimred.fit(X_sparse)
        assert False
    except TypeError:
        assert True

def test_mnist_data():
    # loading modified mnist dataset
    # It contains 2000 labeled images of each digit 0 and 1. Images are 28x28 pixels
    # Classes: 2 (digits 0 and 1)
    # Samples per class: 2000 samples per class
    # Samples total: 4000
    # Dimensionality: 784 (28 x 28 pixels images)
    # Features: integers calues from 0 to 255 (Pixel Grey color)
    mnist_df = pd.read_csv(MY_DATA_PATH_MNIST)
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

    dimred = DimRed(n_components = .90) # n_components = .90 means that scikit-learn will choose the minimum number of principal components such that 90% of the variance is retained.
    dimred.fit(X)

    mnist_dimensions_before_pca = len(pixel_colnames)
    mnist_dimensions_after_pca = dimred.n_components_
    print('\n[test_mnist_data] - Number of dimensions before PCA: ' + str(mnist_dimensions_before_pca))
    print('[test_mnist_data] - Number of dimensions after PCA: ' + str(mnist_dimensions_after_pca))

    assert(mnist_dimensions_before_pca == 784)
    assert(mnist_dimensions_after_pca == 48)
