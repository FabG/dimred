"""
dimred.py

DimRed is a python package to perform Dimension Reduction using PCA by default and other algorithms.

"""
import numpy as np
from numpy import count_nonzero
import scipy.sparse as sp
from scipy.sparse import csr_matrix, isspmatrix
from sklearn.utils.extmath import svd_flip, stable_cumsum
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD

SPARSITY=0.6 # define the %sparsity of a matrix -  0.6 means 60% of values are 0
N_COMPONENTS = 0.95 # default values for returning components using a variance of 95%

class DimRed():
    """
    Linear dimensionality reduction class
    """

    def __init__(self, algo='pca', n_components=N_COMPONENTS, random_int=None):
        """
        Initialize DimRed with user-defined parameters, defaulting to PCA algorithm

        Parameters
        ----------
        algo: Algorithm used to perform Principal Component analysis
            Values:
                "pca" (default) - use scikit learn decomposition.PCA() function based of SVD "as-is"
                    as a pass-through. Results should be the same as if calling decomposiiton.PCA()
                "dimred_svd" - use Singular Value Decomposition for PCA with numpy (internally built)
                "dimred_evd" - use Eigen Value Decomposition for PCA with numpy (internally built)
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
         random : int optional
                    Random state
                    Pass an int for reproducible results across multiple function calls.
        """

        # Store in object
        self.n_components = n_components
        self.algo = algo
        self.sp_issparse = False
        self.issparse = False
        self.random_int = random_int


    def fit_transform(self, X):
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
        model = self._fit_transform(X)
        return (model)


    def _fit_transform(self, X):
        """
        Dispatch to the right submethod depending on the chosen solver
            and apply the dimensionality reduction on X
        """

        # Preprocessing
        X_centered, n_samples, n_features = self._preprocess(X)

        # Dispath to right PCA algorithm
        if self.sp_issparse:
            print('[dimred]: X is sparse - using TruncatedSVD')
            X_dimred = self._pca_truncated_svd(X)

        if self.algo == 'pca':
            print('[dimred]: using sklearn PCA')
            X_dimred = self._pca(X_centered)

        if self.algo == 'dimred_svd':
            print('[dimred]: using DimRed implementation of SVD for PCA')
            X_dimred = self._dimred_pca_svd(X_centered)

        if self.algo == 'dimred_evd':
            print('[dimred]: using DimRed implementation of EVD for PCA')
            X_dimred = self._dimred_pca_evd(X_centered)

        return(X_dimred)


    def _pca_truncated_svd(self, X):
        """
        Use ScikitLearn TruncatedSVD
        Dimensionality reduction using truncated SVD (aka LSA).
        This transformer performs linear dimensionality reduction by means of
        truncated singular value decomposition (SVD). Contrary to PCA, this
        estimator does not center the data before computing the singular value
        decomposition. This means it can work with sparse matrices
        efficiently.
        """

        pca = TruncatedSVD(n_components=self.n_components, random_state=self.random_int)
        X_transf = pca.fit_transform(X)

        return(X_transf)

    def _pca(self, X):
        """
        Use ScikitLearn PCA
        Linear dimensionality reduction using Singular Value Decomposition of the
        data to project it to a lower dimensional space. The input data is centered
        but not scaled for each feature before applying the SVD.
        It uses the LAPACK implementation of the full SVD or a randomized truncated
        SVD by the method of Halko et al. 2009, depending on the shape of the input
        data and the number of components to extract.
        It can also use the scipy.sparse.linalg ARPACK implementation of the
        truncated SVD.
        Notice that this class does not support sparse input. See
        `TruncatedSVD` for an alternative with sparse data.
        """

        pca = PCA(n_components=self.n_components, random_state=self.random_int)
        X_transf = pca.fit_transform(X)

        # Postprocessing
        X_transf = self._postprocess_pca(X_transf, pca)

        return(X_transf)


    def _dimred_pca_svd(self, X_centered):
        """
        Compute SVD based PCA and return Principal Components
        Principal component analysis using SVD: Singular Value Decomposition
        X . V = U . S ==> X = U.S.Vt
        Vt is the matrix that rotate the data from one basis to another
        Note: SVD is a factorization of a real or complex matrix that generalizes
         the eigendecomposition of a square normal matrix to any
         mxn  matrix via an extension of the polar decomposition.

        """
        n_samples, n_features = X_centered.shape

        # SVD
        # full_matricesbool = False => U and Vh are of shape (M, K) and (K, N), where K = min(M, N).
        U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)
        components_ = Vt

        # Get variance explained by singular values
        explained_variance_ = (Sigma ** 2) / (n_samples - 1)

        # Postprocess the number of components required
        X_centered = self._postprocess_dimred_pcasvd(X_centered, Sigma, components_, explained_variance_)

        # Return principal components and eigenvalues to calculate the portion of sample variance explained
        return U, Sigma, Vt



    def _dimred_pca_evd(self, X_centered):
        """
        Compute EVD based PCA and return Principal Components
            and eigenvalues sorted from high to low
        """
        # Build Covariance Matrix
        X_cov = DimRed._cov(X_centered)

        # EVD
        E_vals, E_vecs = DimRed._eigen_sorted(X_cov)
        U = np.dot(X_centered, E_vecs)

        # Return principal components and eigenvalues to calculate the portion of sample variance explained
        return U, E_vals


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


    def _preprocess(self, X):
        """
        Preprocessing
        """

        # Check if input matrix is sparse
        # scipy.sparse defines a number of optimized sparse objects and issparse
        # determines if the insput is ot type scipy.sparse matrix object
        # To ntoe some matrixes can still be sparsed but not of that optimized object type
        if sp.issparse(X): # compressed format of type scipy.sparse
            self.sp_issparse = True
            self.issparse = True
            self.sparsity = 1.0 - csr_matrix.getnnz(X) / (X.shape[0] * X.shape[1])
            print('[dimred]: X is of type scipy.isparse')

        else: # non compressed
            self.sparsity = 1.0 - count_nonzero(X) / X.size
            if self.sparsity > SPARSITY:
                self.issparse = True

        if self.issparse: print('[dimred]: X has a sparsity of: {}'.format(self.sparsity))
        else: print('[dimred]: X is not sparse')

        n_samples, n_features = X.shape
        self.n_samples_, self.n_features_ = n_samples, n_features
        print('[dimred]: X has {} observations and {} features'.format(n_samples, n_features))

        if n_features == 1:
            raise ValueError("Number of features {} implies there is not dimensionality reduction that is possible".format(n_features))

        if self.n_components > n_features:
            print('[dimred]: Warning - Number of components {} cannot be higher than number of features {}'.format(self.n_components, n_features))
            print('[dimred]: Warning - n_components will be set instead to: {}'.format(n_features - 1))
            self.n_components = n_features - 1

        # Center X
        return DimRed._center(X), n_samples, n_features


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


    def _postprocess_pca(self, X, pca):
        """
        Postprocessing for sklearn PCA
        """
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.singular_values_ = pca.singular_values_
        self.n_components_ = pca.n_components_
        self.noise_variance_ = pca.noise_variance_

        return X


    def _postprocess_dimred_pcasvd(self, X, Sigma, components_, explained_variance_):
        """
        Postprocessing for PCA SVD
        """
        n_samples, n_features = X.shape

        if self.n_components is None:
            self.n_components = X.shape[1] - 1

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
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        return X
