"""
dimred.py

DimRed is a python package to perform Dimension Reduction using PCA by default and other algorithms.

"""
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import svd_flip, stable_cumsum

class DimRed():
    """
    Linear dimensionality reduction class
    """

    def __init__(self, algo='pca_svd', n_components=None):
        """
        Initialize DimRed with user-defined parameters, defaulting to PCA algorithm

        Parameters
        ----------
        algo: Algorithm used to perform Principal Component analysis
            Values:
                "pca_svd" (default) - use Singular Value Decomposition
                "pca_evd" - use Eigen Value Decomposition
                    (1) Compute the covariance matrix of the data
                    (2) Compute the eigen values and vectors of this covariance matrix
                    (3) Use the eigen values and vectors to select only the most important feature vectors and then transform your data onto those vectors for reduced dimensionality!

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
        Fit the model with X

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

        if self.algo == 'pca_sklearn':
            model_pca = PCA(n_components=self.n_components)
            model_pca.fit(X)
            self.n_components_ = model_pca.n_components_
            self.explained_variance_ratio_ = model_pca.explained_variance_ratio_
            self.singular_values_ = model_pca.singular_values_
            self.percent_explained_variance = model_pca.explained_variance_ratio_.cumsum()

        if self.algo == 'pca_svd':
            return self._fit_pca_svd(X, self.n_components)


        return(self)


    def _fit_pca_svd(self, X, n_components):
        """
        Compute SVD based PCA and return Principal Components
        """
        # Center X
        X_centered = DimRed._center(X)

        # SVD => X = U x Sigma x Vt
        U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        components_ = Vt

        # Get variance explained by singular values
        n_samples, n_features = X.shape        
        explained_variance_ = (Sigma ** 2) / (n_samples - 1)

        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = Sigma.copy()  # Store the singular values.

        # Postprocess the number of components required
        if 0 < n_components < 1.0:
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components,
                                           side='right') + 1

        # Compute noise covariance using Probabilistic PCA model
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        return U, Sigma, Vt



    def pca_evd(X):
        """
        Compute EVD based PCA and return Principal Components
            and eigenvalues sorted from high to low
        """
        X_cov = _cov(X)
        e_vals, e_vecs = _eigen_sorted(X_cov)

        return X.dot(e_vecs), e_vals


    def _center(X):
        """
        Center a matrix
        """
        n_samples, n_features = X.shape
        x_mean_vec = np.mean(X, axis=0)
        X_centered = X - x_mean_vec

        return X_centered


    def _cov(X):
        """
        Compute a Covariance matrix
        """
        n_samples, n_features = X.shape
        x_mean_vec = np.mean(X, axis=0)
        X_centered = X - x_mean_vec
        X_cov = X_centered.T.dot(X_centered) / (n_samples - 1)

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

        return eig_vals[idx], eig_vecs[:, idx]
