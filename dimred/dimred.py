"""
dimred.py

DimRed is a python package to perform Dimension Reduction using PCA by default and other algorithms.

"""
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DimRed():
    """
    DimRed module
    """

    def __init__(self, algo='pca', n_components=None):
        """
        Initialize DimRed with user-defined parameters, defaulting to PCA algorithm

        Parameters
        ----------
        algo": Algorithm - default is 'pca' (Principal Component analysis)
            More algorithms will be added to this package over time such as TruncatedSVD.
        n_components : Number of components to keep.
            Missing Value => All components are kept.
            Values > 0 are the number of Top components.
                Ex: n_components = 3 => returns Top 3 principal components
            Values < 0 are the components that cover at least the percentage of variance.
                Ex: n_components = 0.85 => returns all components that cover at least 85% of variance.
        """


        # Store in object
        self.n_components = n_components
        self.algo = algo

    def fit(self, X):
        """
        Fit PCA on data

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        model = self._fit(X)
        return (model)

    def _fit(self, X):
        """
        Dispatch to the right submethod depending on the chosen solver
            and apply the dimensionality reduction on X
        """

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if sp.issparse(X):
            raise TypeError('PCA does not support sparse input. See TruncatedSVD for a possible alternative.')

        if self.algo == 'pca':
            model_pca = PCA(n_components=self.n_components)
            model_pca.fit(X)
            self.n_components_ = model_pca.n_components_
            self.explained_variance_ratio_ = model_pca.explained_variance_ratio_
            self.singular_values_ = model_pca.singular_values_
            self.percent_explained_variance = model_pca.explained_variance_ratio_.cumsum()

        return(self)


    def pca_evd(X):
        """
        Compute EVD based PCA and return Principal Components
            and eigenvalues sorted from high to low
        """
        X_cov = _cov(X)
        e_vecs, e_vals = _eigen_sorted(X_cov)

        return X.dot(e_vecs), e_vals


    def _cov(X):
        """
        Compute a Covariance matrix
        """
        n, p = X.shape
        x_mean_vec = np.mean(X, axis=0)
        X_centered = X - x_mean_vec
        X_cov = X_centered.T.dot(X_centered) / (n-1)

        return X_cov


    def _eigen_sorted(X_cov):
        """
        Compute the eigen values and vectors using numpy
            and return the eigenvalue and eigenvectors
            sorted based on eigenvalue from high to low
        """
        # Compute the eigen values and vectors using numpy
        eig_vals, eig_vecs = np.linalg.eig(X_cov)

        # Sort the eigenvalue and eigenvector from high to low
        idx = eig_vals.argsort()[::-1]

        return eig_vals[idx], eig_vals[:, idx]
