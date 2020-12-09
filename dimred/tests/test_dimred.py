import os
import numpy as np
from numpy import count_nonzero
import pandas as pd
from dimred import DimRed
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, isspmatrix
from sklearn import datasets

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


def test_iris_data():
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    dimred = DimRed(n_components=2)
    X_pca = dimred.fit(X)

    explained_variance_ratio = dimred.explained_variance_ratio_
    singular_values = dimred.singular_values_

    print('\n[test_iris_data] - Explained Variance ratio: {}'.format(explained_variance_ratio))
    print('[test_iris_data] - Singular Values: {}'.format(singular_values))

    assert(explained_variance_ratio[0] == 0.9246187232017271)
    assert(explained_variance_ratio[1] == 0.05306648311706782)
    assert(singular_values[0] == 25.099960442183864)
    assert(singular_values[1] == 6.013147382308733)

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

def test_center():
    X = np.array([[0, 3, 4], [1, 2, 4], [3, 4, 5]])
    X_center_ref = np.array([[-1.33333333, 0., -0.33333333],[-0.33333333, -1., -0.33333333],[1.66666667, 1., 0.66666667]])
    X_center = DimRed._center(X)
    print(X)

    print('\n[test_center] - Checking Matrix Center: _center(X)')
    assert(np.allclose(X_center, X_center_ref))


def test_covariance():
    X = np.array([[0, 3, 4], [1, 2, 4], [3, 4, 5]])
    X_cov_ref = np.array([[2.3333333333333335, 1., 0.8333333333333334],[1. , 1., 0.5], [0.8333333333333334, 0.5, 0.3333333333333333]])
    X_cov = DimRed._cov(X)
    print(X)

    print('\n[test_covariance] - Checking Matrix Covariance: _cov(X)')
    assert(np.array_equal(X_cov, X_cov_ref))


def test_eigen():
    X_cov_ref = np.array([[2.3333333333333335, 1., 0.8333333333333334],[1. , 1., 0.5], [0.8333333333333334, 0.5, 0.3333333333333333]])

    X_eig_vecs_ref = np.array([[-0.83234965, -0.50163583, -0.23570226],
                            [-0.45180545,  0.86041634, -0.23570226],
                            [-0.32103877,  0.08969513,  0.94280904]])
    X_eig_vals_ref = np.array([ 3.19755880e+00,  4.69107871e-01, -3.13055232e-18])

    X_eig_vals, X_eig_vecs = DimRed._eigen_sorted(X_cov_ref)

    print('\n[test_eigen] - Checking Eigen Sorted _eigen_sorted(X_cov)')
    assert(np.allclose(X_eig_vals, X_eig_vals_ref))  # avoiding rounding float errors
    assert(np.allclose(X_eig_vecs, X_eig_vecs_ref))  # avoiding rounding float errors
