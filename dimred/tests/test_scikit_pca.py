import os
import numpy as np
from numpy import count_nonzero
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits, make_friedman1
from scipy.sparse import csr_matrix, isspmatrix
from scipy.sparse import random as sparse_random


def test_np_array():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=2)
    pca.fit(X)

    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_

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

    pca = PCA(n_components=1)
    try:
        pca.fit(X_sparse)
        assert True
        explained_variance_ratio = pca.explained_variance_ratio_
        singular_values = pca.singular_values_

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

    pca = PCA(n_components=2)
    try:
        pca.fit(X_sparse)
        assert False
    except TypeError:
        assert True

def test_iris_data():
    iris = load_iris()

    X = iris.data
    y = iris.target

    pca = PCA(n_components=2)
    X_pca = pca.fit(X)

    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_

    assert(explained_variance_ratio[0] == 0.9246187232017271)
    assert(explained_variance_ratio[1] == 0.05306648311706782)
    assert(singular_values[0] == 25.099960442183864)
    assert(singular_values[1] == 6.013147382308733)


def test_iris_data_transform():
    iris = load_iris()

    X = iris.data
    y = iris.target

    pca = PCA(n_components=2)
    X_pca = pca.fit(X).transform(X)
    X_pca2 = pca.fit_transform(X)

    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_

    assert(explained_variance_ratio[0] == 0.9246187232017271)
    assert(explained_variance_ratio[1] == 0.05306648311706782)
    assert(singular_values[0] == 25.099960442183864)
    assert(singular_values[1] == 6.013147382308733)

    #assert(np.array_equal(X_pca, X_pca2))
    assert(np.allclose(X_pca, X_pca2))  # avoiding rounding float errors


def test_mnist_data_pca():
    # Load and return the digits dataset (classification).
    # Each datapoint is a 8x8 image of a digit.
    # Dimensionality = 64
    # Features = integers 0-16
    # Observations = 1797
    digits = load_digits(as_frame=True)
    X = digits.data
    y = digits.target
    pixel_colnames = digits.feature_names

    # PCA is sensitive to the scale of the features.
    # We can standardize your data onto unit scale (mean = 0 and variance = 1) by using Scikit-Learn's StandardScaler.
    scaler = StandardScaler()
    scaler.fit(X)

    pca = PCA(n_components = .90) # n_components = .90 means that scikit-learn will choose the minimum number of principal components such that 90% of the variance is retained.
    pca.fit(X)

    mnist_dimensions_before_pca = len(pixel_colnames)
    mnist_dimensions_after_pca = pca.n_components_

    assert(mnist_dimensions_before_pca == 64)
    assert(mnist_dimensions_after_pca == 21)


def test_sparse_pca():
    X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
    transformer = SparsePCA(n_components=5, random_state=0)
    transformer.fit(X)
    X_transformed = transformer.transform(X)
    # X.shape = (200, 30) => reduced to X_transformed.shape = (200,5)
    assert(X.shape == (200, 30))
    assert(X_transformed.shape == (200, 5))

    assert(np.mean(transformer.components_ == 0))
    assert(np.allclose(transformer.mean_, X.mean(axis=0)))

def test_truncated_svd():
    X = sparse_random(100, 100, density=0.01, format='csr', random_state=42)
    explained_variance_ratio_ref = np.array([0.06461231, 0.06338995, 0.06394725, 0.05351761, 0.04064443])
    explained_variance_ratio_sum_ref = 0.28611154708177045
    singular_values_ref = np.array([1.5536061 , 1.51212835, 1.51050701, 1.37044879, 1.19768771])

    svd = TruncatedSVD(n_components=5, random_state=42)
    X_transformed = svd.fit_transform(X)

    assert(np.allclose(svd.explained_variance_ratio_, explained_variance_ratio_ref))  # avoiding rounding float errors
    assert(svd.explained_variance_ratio_.sum() == explained_variance_ratio_sum_ref)
    assert(np.allclose(svd.singular_values_, singular_values_ref))  # avoiding rounding float errors

    assert(X.shape == (100, 100))
    assert(X_transformed.shape == (100, 5))
