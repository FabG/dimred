import os
import numpy as np
from numpy import count_nonzero
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman1
from scipy.sparse import csr_matrix, isspmatrix
from scipy.sparse import random as sparse_random
from sklearn import datasets

# Set up absolute path to unit test files
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
#MY_DATA_PATH = os.path.join(THIS_DIR, os.pardir, 'data/data.csv')
MY_DATA_PATH_MNIST = os.path.join(THIS_DIR, 'data/mnist_only_0_1.csv')
MY_DATA_PATH_IRIS = os.path.join(THIS_DIR, 'data/iris_data.csv')
print('MY_DATA_PATH_MNIST = {}'.format(MY_DATA_PATH_MNIST))

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


def test_np_array_sparse_noncsr():
    # create sparse matrix
    X_sparse = np.array([[1,0,0,0,0,0], [0,0,2,0,0,0], [0,0,0,2,0,0]])

    # calculate sparsity
    sparsity = 1.0 - count_nonzero(X_sparse) / X_sparse.size
    # The above array has 0.833 sparsity (meaning 83.3% of its values are 0)
    print('\n[test_np_array_sparse_noncsr] - Checking no exception for sparsity of: {:.2f}'.format(sparsity))

    pca = PCA(n_components=1)
    try:
        pca.fit(X_sparse)
        assert True
        explained_variance_ratio = pca.explained_variance_ratio_
        singular_values = pca.singular_values_

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
    print('\n[test_np_array_sparse_csr] - Checking compressed sparse exception for sparsity of: {:.2f}'.format(sparsity))

    pca = PCA(n_components=2)
    try:
        pca.fit(X_sparse)
        assert False
    except TypeError:
        assert True

def test_iris_data():
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    pca = PCA(n_components=2)
    X_pca = pca.fit(X)

    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_

    print('\n[test_iris_data] - Explained Variance ratio: {}'.format(explained_variance_ratio))
    print('[test_iris_data] - Singular Values: {}'.format(singular_values))

    assert(explained_variance_ratio[0] == 0.9246187232017271)
    assert(explained_variance_ratio[1] == 0.05306648311706782)
    assert(singular_values[0] == 25.099960442183864)
    assert(singular_values[1] == 6.013147382308733)


def test_iris_data_transform():
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    pca = PCA(n_components=2)
    X_pca = pca.fit(X).transform(X)
    X_pca2 = pca.fit_transform(X)

    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_

    print('\n[test_iris_data_transform] - Explained Variance ratio: {}'.format(explained_variance_ratio))
    print('[test_iris_data_transform] - Singular Values: {}'.format(singular_values))

    assert(explained_variance_ratio[0] == 0.9246187232017271)
    assert(explained_variance_ratio[1] == 0.05306648311706782)
    assert(singular_values[0] == 25.099960442183864)
    assert(singular_values[1] == 6.013147382308733)

    #assert(np.array_equal(X_pca, X_pca2))
    assert(np.allclose(X_pca, X_pca2))  # avoiding rounding float errors


def test_mnist_data_pca():
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

    pca = PCA(n_components = .90) # n_components = .90 means that scikit-learn will choose the minimum number of principal components such that 90% of the variance is retained.
    pca.fit(X)

    mnist_dimensions_before_pca = len(pixel_colnames)
    mnist_dimensions_after_pca = pca.n_components_
    print('\n[test_mnist_data] - Number of dimensions before PCA: ' + str(mnist_dimensions_before_pca))
    print('[test_mnist_data] - Number of dimensions after PCA: ' + str(mnist_dimensions_after_pca))

    assert(mnist_dimensions_before_pca == 784)
    assert(mnist_dimensions_after_pca == 48)


def test_sparse_pca():
    X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
    transformer = SparsePCA(n_components=5, random_state=0)
    transformer.fit(X)
    X_transformed = transformer.transform(X)
    # X.shape = (200, 30) => reduced to X_transformed.shape = (200,5)
    assert (X.shape[0] == 200)
    assert (X.shape[1] == 30)
    assert (X_transformed.shape[0] == 200)
    assert (X_transformed.shape[1] == 5)

    assert (np.mean(transformer.components_ == 0))
    assert (np.allclose(transformer.mean_, X.mean(axis=0)))

def test_truncated_svd():
    print('\n[test_truncated_svd]')
    X = sparse_random(100, 100, density=0.01, format='csr', random_state=42)
    explained_variance_ratio_ref = np.array([0.06461231, 0.06338995, 0.06394725, 0.05351761, 0.04064443])
    explained_variance_ratio_sum_ref = 0.28611154708177045
    singular_values_ref = np.array([1.5536061 , 1.51212835, 1.51050701, 1.37044879, 1.19768771])

    svd = TruncatedSVD(n_components=5, random_state=42)
    X_transformed = svd.fit_transform(X)

    assert(np.allclose(svd.explained_variance_ratio_, explained_variance_ratio_ref))  # avoiding rounding float errors
    assert(svd.explained_variance_ratio_.sum() == explained_variance_ratio_sum_ref)
    assert(np.allclose(svd.singular_values_, singular_values_ref))  # avoiding rounding float errors

    assert (X.shape[0] == 100)
    assert (X.shape[1] == 100)
    assert (X_transformed.shape[0] == 100)
    assert (X_transformed.shape[1] == 5)
