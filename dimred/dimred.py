"""
dimred.py

DimRed is a python package to perform Dimension Reduction using PCA by default and other algorithms.

"""
import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import svd_flip, stable_cumsum

class DimRed():
    """
    Linear dimensionality reduction class
    """

    def __init__(self, algo='pca_svd_full', n_components=0.95):
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

        if self.algo == 'pca_svd_full':
            return self._fit_pca_svd_full(X)

        if self.algo == 'pca_svd':
            return self._fit_pca_svd(X)

        if self.algo == 'pca_evd':
            return self._fit_pca_evd(X)


        return(self)


    def _fit_pca_svd_full(self, X):
        """
        Compute SVD based PCA and return Principal Components
        Principal component analysis using SVD: Singular Value Decomposition
        X . V = U . S ==> X = U.S.Vt
        Vt is the matrix that rotate the data from one basis to another
        """

        # Pre-processing
        X, n_samples, n_features = self._preprocess(X)

        # Center X
        X_centered = DimRed._center(X)

        # SVD => X = U x Sigma x Vt
        # full_matricesbool = False => U and Vh are of shape (M, K) and (K, N), where K = min(M, N).
        U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        components_ = Vt

        # Get variance explained by singular values
        explained_variance_ = (Sigma ** 2) / (n_samples - 1)

        # Postprocess the number of components required
        X = self._postprocess(X, Sigma, components_, explained_variance_)

        return U, Sigma, Vt


    def _fit_pca_svd(self, X):
        """
        Compute SVD based PCA and return Principal Components
        Principal component analysis using SVD: Singular Value Decomposition
        X . V = U . S ==> X = U.S.Vt
        Vt is the matrix that rotate the data from one basis to another
        Note: SVD is a factorization of a real or complex matrix that generalizes
         the eigendecomposition of a square normal matrix to any
         mxn  matrix via an extension of the polar decomposition.

        """
        # Pre-processing
        X, n_samples, n_features = self._preprocess(X)

        # Center X
        X_centered = DimRed._center(X)

        # SVD
        U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        # Get variance explained by singular values
        explained_variance_ = (Sigma ** 2) / (n_samples - 1)

        # Return principal components and eigenvalues to calculate the portion of sample variance explained
        return U, Sigma, Vt



    def _fit_pca_evd(self, X):
        """
        Compute EVD based PCA and return Principal Components
            and eigenvalues sorted from high to low
        """
        # Center X and build Covariance Matrix
        X_cov = DimRed._cov(X)

        # EVD
        E_vals, E_vecs = DimRed._eigen_sorted(X_cov)
        U = np.dot(X, E_vecs)

        # Return principal components and eigenvalues to calculate the portion of sample variance explained
        return U, E_vals


    def _postprocess(self, X, Sigma, components_, explained_variance_):
        """
        Postprocessing for PCA SVD
        """
        n_features, n_samples = X.shape
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = Sigma.copy()  # Store the singular values.

        n_components = self.n_components
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

        return X

    def _preprocess(self, X):
        """
        Preprocessing
        """

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if sp.issparse(X):
            raise TypeError('PCA does not support sparse input. See TruncatedSVD for a possible alternative.')

        if self.n_components is None:
            self.n_components = X.shape[1] - 1

        n_samples, n_features = X.shape

        return(X, n_samples, n_features)


    def _center(X):
        """
        Center a matrix
        """
        x_mean_vec = np.mean(X, axis=0)
        X_centered = X - x_mean_vec

        return X_centered


    def _cov(X):
        """
        Compute a Covariance matrix
        """
        n_samples, n_features = X.shape
        X_centered = DimRed._center(X)
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
