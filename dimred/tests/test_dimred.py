import os
import numpy as np
from numpy import count_nonzero
import pandas as pd
from dimred import DimRed
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, isspmatrix
from sklearn.utils.extmath import svd_flip, stable_cumsum
from sklearn import datasets

# Set up absolute path to unit test files
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
#MY_DATA_PATH = os.path.join(THIS_DIR, os.pardir, 'data/data.csv')
MY_DATA_PATH_MNIST = os.path.join(THIS_DIR, 'data/mnist_only_0_1.csv')
MY_DATA_PATH_IRIS = os.path.join(THIS_DIR, 'data/iris_data.csv')
print('MY_DATA_PATH_MNIST = {}'.format(MY_DATA_PATH_MNIST))


def test_init():
    dimred = DimRed()
    dimred2 = DimRed(algo='pca_svd')
    dimred3 = DimRed(algo='pca_evd', n_components=3)
    assert (dimred.n_components == 0.95)
    assert (dimred.algo == 'pca_svd')
    assert (dimred2.n_components == 0.95)
    assert (dimred2.algo == 'pca_svd')
    assert (dimred3.n_components == 3)
    assert (dimred3.algo == 'pca_evd')


def test_np_array_2_components():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    dimred = DimRed(n_components=2)
    model = dimred.fit_transform(X)
    explained_variance_ratio = dimred.explained_variance_ratio_

    print('\n[test_np_array] - Explained Variance ratio: {}'.format(explained_variance_ratio))

    assert(explained_variance_ratio[0] == 0.9924428900898052)
    assert(explained_variance_ratio[1] == 0.007557109910194766)

def test_np_array_default_components():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    dimred = DimRed()  #0.95 default
    dimred2 = DimRed(n_components=0.95)  #0.95 default
    model = dimred.fit_transform(X)
    model2 = dimred.fit_transform(X)
    explained_variance_ratio = dimred.explained_variance_ratio_
    explained_variance_ratio2 = dimred.explained_variance_ratio_

    print('\n[test_np_array] - Explained Variance ratio: {}'.format(explained_variance_ratio))

    assert(explained_variance_ratio[0] == 0.9924428900898052)
    assert(explained_variance_ratio2[0] == 0.9924428900898052)

def test_np_array_sparse_noncsr():
    # create sparse matrix
    X_sparse = np.array([[1,0,0,0,0,0], [0,0,2,0,0,0], [0,0,0,2,0,0]])

    # calculate sparsity
    sparsity = 1.0 - count_nonzero(X_sparse) / X_sparse.size
    # The above array has 0.833 sparsity (meaning 83.3% of its values are 0)
    print('\n[test_np_array_sparse_noncsr] - Checking no exception for sparsity of: {}'.format(sparsity))

    dimred = DimRed(n_components=1)
    try:
        dimred.fit_transform(X_sparse)
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
        dimred.fit_transform(X_sparse)
        assert False
    except TypeError:
        assert True


def test_iris_data():
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    dimred = DimRed(n_components=2)
    X_pca = dimred.fit_transform(X)

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
    dimred.fit_transform(X)

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


def test_preprocess():
    X = np.array([[0, 3, 4], [1, 2, 4], [3, 4, 5]])
    X_center_ref = np.array([[-1.33333333, 0., -0.33333333],[-0.33333333, -1., -0.33333333],[1.66666667, 1., 0.66666667]])

    dimred = DimRed()
    X = dimred._preprocess(X)
    print('\n[test_preprocess] - Checking Matrix Center amd n_components: _preprocess(X)')
    assert(np.allclose(X, X_center_ref))

def test_eigen_sorted():
    X_cov_ref = np.array([[2.3333333333333335, 1., 0.8333333333333334],[1. , 1., 0.5], [0.8333333333333334, 0.5, 0.3333333333333333]])

    X_eig_vecs_ref = np.array([[-0.83234965, -0.50163583, -0.23570226],
                            [-0.45180545,  0.86041634, -0.23570226],
                            [-0.32103877,  0.08969513,  0.94280904]])
    X_eig_vals_ref = np.array([ 3.19755880e+00,  4.69107871e-01, -3.13055232e-18])

    X_eig_vals, X_eig_vecs = DimRed._eigen_sorted(X_cov_ref)

    print('\n[test_eigen_sorted] - Checking Eigen Sorted: _eigen_sorted(X_cov)')
    assert(np.allclose(X_eig_vals, X_eig_vals_ref))  # avoiding rounding float errors
    assert(np.allclose(X_eig_vecs, X_eig_vecs_ref))  # avoiding rounding float errors


def test_pca_evd():
    X = np.array([[0, 3, 4], [1, 2, 4], [3, 4, 5]])
    X_vec_ref = np.array([[-2.63957145,  2.94002954,  3.06412939],
                        [-3.02011565,  1.57797737,  3.06412939],
                        [-5.90946462,  2.38523353,  3.06412939]])
    e_vals_ref = np.array([ 3.19755880e+00,  4.69107871e-01, -3.13055232e-18])

    dimred = DimRed()  #0.95 default

    # Covariance (implemented by _cov())
    n_samples, n_features = X.shape
    x_mean_vec = np.mean(X, axis=0)
    X_centered = X - x_mean_vec
    X_cov = X_centered.T.dot(X_centered) / (n_samples - 1)

    # Eigen values (implemented by _eigen_sorted)
    eig_vals, eig_vecs = np.linalg.eig(X_cov)
    idx = eig_vals.argsort()[::-1]
    e_vals, e_vecs = eig_vals[idx], eig_vecs[:, idx]

    X_vecs, e_vals = X.dot(e_vecs), e_vals

    X_vecs_fct, e_vals_fct = dimred._pca_evd(X)

    print('\n[test_pca_evd] - Checking Eigen vectors and values: _fit_pca_evd(X)')
    assert(np.allclose(X_vecs, X_vec_ref))  # avoiding rounding float errors
    assert(np.allclose(X_vecs_fct, X_vec_ref))  # avoiding rounding float errors
    assert(np.allclose(e_vals, e_vals_ref))  # avoiding rounding float errors
    assert(np.allclose(e_vals_fct, e_vals_ref))  # avoiding rounding float errors

def test_pca_svd():
    X = np.array([[0, 3, 4], [1, 2, 4], [3, 4, 5]])
    U_ref = np.array([[-0.48117093, -0.65965234,  0.57735027],
                            [-0.33069022,  0.74653242,  0.57735027],
                            [ 0.81186114, -0.08688008,  0.57735027]])
    Vt_ref = np.array([[ 0.83234965,  0.45180545,  0.32103877],
                        [ 0.50163583, -0.86041634, -0.08969513],
                        [-0.23570226, -0.23570226,  0.94280904]])
    Sigma_ref = np.array([2.52885697e+00, 9.68615374e-01, 5.82986245e-16])

    dimred = DimRed()  #0.95 default

    # Center matrix
    n_samples, n_features = X.shape
    x_mean_vec = np.mean(X, axis=0)
    X_centered = X - x_mean_vec

    # SVD - manual
    U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)
    U, Vt = svd_flip(U, Vt)
    components_ = Vt
    explained_variance_ = (Sigma ** 2) / (n_samples - 1)

    # post preprocessing
    X_centered = dimred._postprocess(X_centered, Sigma, components_, explained_variance_)

    # SVD - function
    U2, Sigma2, Vt2 = dimred._pca_svd(X_centered)

    print('\n[test_pca_svd] - Checking Eigen vectors and values: _fit_pca_svd(X)')
    assert(np.allclose(U, U_ref))  # avoiding rounding float errors
    assert(np.allclose(U2, U_ref))  # avoiding rounding float errors
    assert(np.allclose(Vt, Vt_ref))  # avoiding rounding float errors
    assert(np.allclose(Vt, Vt_ref))  # avoiding rounding float errors
    assert(np.allclose(Sigma, Sigma_ref))  # avoiding rounding float errors
    assert(np.allclose(Sigma2, Sigma_ref))  # avoiding rounding float errors
