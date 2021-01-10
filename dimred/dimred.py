"""
dimred.py

DimRed is a python package to perform Dimension Reduction
It uses automatically different algorithms based on input data (sparse or not)
and/or based on user's input parameter.
Some algorithms come from sklearn: PCA, SparsePCA, TruncatedSVD
Som others are internally built in numpy to perform PCA with: EVD, SVD

"""
import numpy as np
from numpy import count_nonzero
import scipy.sparse as sp
from scipy.sparse import csr_matrix, isspmatrix
from sklearn.utils.extmath import svd_flip, stable_cumsum
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from logging.handlers import RotatingFileHandler

SPARSITY = 0.6      # define the %sparsity of a matrix -  0.6 means 60% of values are 0
N_COMPONENTS = 0.95 # default values for returning components using a variance of 95%
DEFAULT_PCA_ALGO = 'sklearn_pca'
DEFAULT_TITLE = 'DimRed Plot'
DEFAULT_FIG_SIZE=(8, 6)
LOG_FILE='dimred.log'
LOG_LEVEL=logging.INFO  #DEBUG, INFO, WARNING, ERROR and CRITICAL
#logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

logging.basicConfig(
        handlers=[RotatingFileHandler(LOG_FILE, maxBytes=1000000, backupCount=10)],
        level=LOG_LEVEL,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')
logger = logging.getLogger()

class DimRed():
    """
    Linear dimensionality reduction class
    """

    def __init__(self, algo='auto', n_components=N_COMPONENTS, random_int=None):
        """
        Initialize DimRed with user-defined parameters, defaulting to PCA algorithm

        Parameters
        ----------
        algo: Algorithm used to perform Principal Component analysis
            Values:
                "auto" - pick the PCA method automatically with PCA SVD being the default
                "sklearn_pca"  - use scikit learn decomposition.PCA() function based of SVD "as-is"
                    as a pass-through. Results should be the same as if calling decomposiiton.PCA()
                "dimred_svd" - use Singular Value Decomposition for PCA with numpy (internally built)
                               this should return the same results as "sklearn_truncated_svd"
                "dimred_evd" - use Eigen Value Decomposition for PCA with numpy (internally built)
                    (1) Compute the covariance matrix of the data
                    (2) Compute the eigen values and vectors of this covariance matrix
                    (3) Use the eigen values and vectors to select only the most important feature vectors and then transform your data onto those vectors for reduced dimensionality!
                "sklearn_truncated_svd" - use scikit learn decomposition.TruncatedSVD()
                       this should return the same results as internally built function "dimred_svd"
                "sklearn_sparse_pca" - use scikit learn decomposition.SparsePCA()
            More algorithms will be added to this package over time such as TruncatedSVD.
        n_components : Number of components to keep.
            Missing Value => we will select PC with 95% explained variance
            Values > 0 are the number of Top components.
                Ex: n_components = 3 => returns Top 3 principal components
            Values < 0 are the components that cover at least the percentage of variance.
                Ex: n_components = 0.85 => returns all components that cover at least 85% of variance.
         random_int: Pass an int for reproducible results across multiple function calls.
            Value: int optional (Random state)
        """

        # Store in object
        logger.info('======> DimRed Initialization')
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
        logger.info('<====== DimRed Completion\n')
        return (model)


    def draw_scatterplot(self, X, y=None, PC=2, title=DEFAULT_TITLE,
                        figsize=DEFAULT_FIG_SIZE, legend=False, dim3=False) :
        """
        Render X as a scatter 2d or 3d plot with 2 or 3 components


        Parameters
        ----------
        X : array-like of shape (n_samples, n_components) to be plotted
        y : array-like of shape (n_samples) to be plotted - labels
        PC : int, default : 2
            Plot the Principal Components, default 2. This also accepts 3
            If 3 is selected, the 3rd component will be used as size of the bubbles
        title : string, default : 'DimRed Plot'
            Adds a title to the chart. Pass empty '' if you prefer no title
        figsize : 2-tuple of floats (float, float), optional, default: (10,8)
            Figure dimension (width, height) in inches.
        legend : boolean, default : False
            Displays a legend if set to True
        3d : boolean, default : False
            Displays a map as 3D if there are 3 PCs

        Returns
        -------
        tuple containing (fig, ax)

        """
        # Colormap - uwing `Qualitative` as it changes rapidly
        # see maptplotlib.pyplot cmaps for more info
        #color_list = plt.cm.Set3(np.linspace(0, 1, 12))

        fig, ax = plt.subplots(figsize=figsize, edgecolor='k')
        (axis_title_0, axis_title_1, axis_title_2) = self._get_variance_axis()

        if PC not in (2,3,4):
            raise ValueError("[DimRed] - PC needs to be 2, 3 or 4 to be plotted")

        if dim3:
            ax = Axes3D(fig, elev=-150, azim=110)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

        ax.set_xlabel('Principal Component 1' + axis_title_0)
        ax.set_ylabel('Principal Component 2' + axis_title_1)
        if dim3: ax.set_zlabel('Principal Component 3' + axis_title_2)

        if dim3:
            ax.text2D(0.2, 0.95, title, transform=ax.transAxes)
        else:
            ax.set_title(title)


        if PC == 2:
            scatter = plt.scatter(X[:,0], X[:,1],
                                alpha=0.4, c=y, edgecolor='k',
                                cmap='viridis')

        if PC == 3:
            if dim3:
                # used the 3rd PC as size of the plot 's'
                scatter = ax.scatter(X[:,0], X[:,1], X[:,2],
                                    c=y, alpha=0.4, edgecolor='k',
                                    cmap='viridis', s=40)
            else:
                # used the 3rd PC as size of the plot 's'
                scatter = plt.scatter(X[:,0], X[:,1], s=X[:,2],
                                    c=y, alpha=0.4, edgecolor='k',
                                    cmap='viridis')
                fig.tight_layout()


        if legend:
            if PC in (2,3):
                # produce a legend with the unique colors from the scatter
                legend1 = ax.legend(*scatter.legend_elements(),
                loc="lower left", title="Classes")
                ax.add_artist(legend1)
            if PC == 3:
                if dim3 == False:
                    # produce a legend with a cross section of sizes from the scatter
                    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
                    legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")

        return fig, ax



    def draw_varianceplot(self, title=None, figsize=DEFAULT_FIG_SIZE):
        """
        Render the cumulative variance based of components that match the target variance

        Parameters
        ----------
        title : string, default : 'Cumulative Explained Variance'
            Adds a title to the chart. Pass empty '' if you prefer no title
        figsize : 2-tuple of floats (float, float), optional, default: (10,8)
            Figure dimension (width, height) in inches.

        Returns
        -------
        tuple containing (fig, ax)

        """

        explained_variance = self.explained_variance_
        explained_variance_ratio = self.explained_variance_ratio_

        if 0 < self.n_components < 1:
            total_explained_variance = self.n_components
        else:
            total_explained_variance = np.sum(explained_variance_ratio)

        cumsum_explained_variance = stable_cumsum(self.explained_variance_ratio_)

        # index based on components to plot
        x_index = np.arange(1, len(explained_variance) + 1)

        # Cumulative Variance by component bar chart
        fig, ax = plt.subplots(figsize=figsize, edgecolor='k')
        plt.plot(x_index, cumsum_explained_variance, marker='*', color='royalblue', alpha=0.9, linewidth=1, label='Cumulative explained variance')

        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance %')
        plt.grid(True)

        # setting x ticks to match components
        x_index = np.arange(1, len(explained_variance) + 1)
        ax.set_xticks(x_index)
        # if too many ticklabels, display odd numbers
        if len(explained_variance) > 30:
            for (i, label) in enumerate(ax.xaxis.get_ticklabels()):
                if i % 2 != 0:
                    label.set_visible(False)
        # set our target as X and Y lines
        ax.axvline(self.n_components_, linewidth=1.1, linestyle='--', color='mediumblue')
        ax.axhline(y=total_explained_variance, xmin=0, xmax=1, linewidth=1.1,  linestyle='--', color='mediumblue')

        # Variance by component bar chart
        plt.bar(x_index, explained_variance_ratio, color='mediumseagreen', edgecolor='dimgray', alpha=0.8, label='Explained Variance')

        full_title = ''
        if title is not None: full_title = title + '\n'
        full_title += 'Cumulative Explained Variance\n (' + str(self.n_components_) + ' Principal Components explain [' + str(total_explained_variance * 100)[0:5] + '%] of the variance)'
        plt.title(full_title)

        return fig, ax



    def _get_variance_axis(self):
        """
        return 1 to 3 strings based on variance for scatter plot axis titles
        """
        axis_title_0 = ''
        axis_title_1 = ''
        axis_title_2 = ''
        if len(self.explained_variance_ratio_) > 0:
            axis_title_0 = '\n(' + str(self.explained_variance_ratio_[0]*100)[0:4] + '% explained variance)'
        if len(self.explained_variance_ratio_) > 1:
            axis_title_1 = '\n(' + str(self.explained_variance_ratio_[1]*100)[0:4] + '% explained variance)'
        if len(self.explained_variance_ratio_) > 2:
            axis_title_2 = '\n(' + str(self.explained_variance_ratio_[2]*100)[0:4] + '% explained variance)'

        return (axis_title_0, axis_title_1, axis_title_2)


    def _fit_transform(self, X):
        """
        Dispatch to the right submethod depending on the chosen solver
            and apply the dimensionality reduction on X
        """

        # Preprocessing
        X_centered, n_samples, n_features = self._preprocess(X)

        # Dispath to right PCA algorithm based on input algo or based on data type
        # Check Input Matrix

        if self.algo == 'auto':

            if self.sp_issparse:  # X is of type scipy.sparse
                logger.info('X is sparse and of type scipy.sparse => using sklearn TruncatedSVD')
                self.algo = 'sklearn_truncated_svd'
                X_dimred = self._sklearn_truncated_svd(X)

            elif self.issparse: # X is a sparse matrix with lots of 0 but not of type scipy.sparse
                logger.info('X is sparse => using sklearn SparsePCA')
                self.algo = 'sklearn_sparse_pca'
                #X_dimred = self._sklearn_pca(X_centered)
                # Note - n_components must be an integer for this function
                if self.n_components < 1:
                    self.n_components = X.shape[1] - 1
                    logger.info('SparsePCA can only use n_components as integer - defaulting to {}'.format(self.n_components))

                X_dimred = self._sklearn_sparse_pca(X)

            else: self.algo = DEFAULT_PCA_ALGO  # 'sklearn_pca'

        # Check input algorithm and use default if not available
        if self.algo == 'sklearn_pca':  # default
            logger.info(' => using sklearn PCA')
            X_dimred = self._sklearn_pca(X_centered)

        elif self.algo == 'dimred_svd':
            logger.info(' => using DimRed implementation of SVD for PCA')
            X_dimred = self._dimred_svd(X_centered)

        elif self.algo == 'dimred_evd':
            logger.info('=> using DimRed implementation of EVD for PCA')
            X_dimred = self._dimred_evd(X_centered)

        elif self.algo == 'sklearn_truncated_svd':
            logger.info(' => using sklearn TruncatedSVD')
            X_dimred = self._sklearn_truncated_svd(X)

        elif self.algo == 'sklearn_sparse_pca':
            logger.info(' => using sklearn SparsePCA')
            X_dimred = self._sklearn_sparse_pca(X)

        else:
            logger.error('not able to run')
            raise ValueError("[DimRed] - not able to run")

        return(X_dimred)


    def _sklearn_truncated_svd(self, X):
        """
        Use Scikit Learn TruncatedSVD
        Dimensionality reduction using truncated SVD (aka LSA).
        This transformer performs linear dimensionality reduction by means of
        truncated singular value decomposition (SVD). Contrary to PCA, this
        estimator does not center the data before computing the singular value
        decomposition. This means it can work with sparse matrices
        efficiently.
        """

        pca = TruncatedSVD(n_components=self.n_components, random_state=self.random_int)
        X_transf = pca.fit_transform(X)

        # Postprocessing
        X_transf = self._postprocess_sklearn_truncated_svd(X_transf, pca)
        logger.info('Output Matrix X_transf has {} observations and {} components'.format(X_transf.shape[0], X_transf.shape[1]))

        return(X_transf)


    def _sklearn_pca(self, X):
        """
        Use Scikit Learn PCA
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
        X_transf = self._postprocess_sklearn_pca(X_transf, pca)
        logger.info('Output Matrix X_transf has {} observations and {} components'.format(X_transf.shape[0], X_transf.shape[1]))

        return(X_transf)

    def _sklearn_sparse_pca(self, X):
        """
        Use Scikit Learn Sparse Principal Components Analysis (SparsePCA).
        Finds the set of sparse components that can optimally reconstruct
        the data.  The amount of sparseness is controllable by the coefficient
        of the L1 penalty, given by the parameter alpha.
        """

        pca = SparsePCA(n_components=self.n_components, random_state=self.random_int)
        X_transf = pca.fit_transform(X)

        # Postprocessing
        X_transf = self._postprocess_sklearn_sparsepca(X_transf, pca)
        logger.info('Output Matrix X_transf has {} observations and {} components'.format(X_transf.shape[0], X_transf.shape[1]))

        return(X_transf)


    def _dimred_svd(self, X_centered):
        """
        Compute SVD based PCA and return Principal Components
        Principal component analysis using SVD: Singular Value Decomposition
        X . V = U . S ==> X = U.S.Vt
        Vt is the matrix that rotate the data from one basis to another
        Note: SVD is a factorization of a real or complex matrix that generalizes
         the eigendecomposition of a square normal matrix to any
         mxn  matrix via an extension of the polar decomposition.

        """
        # SVD
        # full_matricesbool = False => U and Vh are of shape (M, K) and (K, N), where K = min(M, N).
        U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        # Postprocess the number of components required
        X_transf = self._postprocess_dimred_pca_svd(U, Sigma, Vt)
        logger.info('n_features_: {}'.format(self.n_features_))
        logger.info('n_samples_: {}'.format(self.n_samples_))
        logger.info('Output Matrix X_transf has {} observations and {} components'.format(X_transf.shape[0], X_transf.shape[1]))

        # Return principal components
        return X_transf


    def _dimred_evd(self, X_centered):
        """
        Compute EVD based PCA and return Principal Components
            and eigenvalues sorted from high to low
        """
        # Build Covariance Matrix
        X_cov = DimRed._cov(X_centered)

        # EVD
        eigen_vals_sorted, eigen_vecs_sorted = DimRed._eigen_sorted(X_cov)
        logger.info('eigen_vals_sorted: \n{}'.format(eigen_vals_sorted))
        logger.info('eigen_vecs_sorted: \n{}'.format(eigen_vecs_sorted))

        # Postprocess the number of components required
        X_transf = self._postprocess_dimred_pca_evd(X_centered, eigen_vals_sorted, eigen_vecs_sorted)
        logger.info('Output Matrix X_transf has {} observations and {} components'.format(X_transf.shape[0], X_transf.shape[1]))

        # Return principal components
        return X_transf



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
        n_samples, n_features = X.shape
        self.n_samples_, self.n_features_ = n_samples, n_features
        logger.info('Input Matrix X has {} observations and {} features'.format(n_samples, n_features))
        logger.info('TEST - self.n_features_: {}'.format(self.n_features_))

        if n_features == 1:
            raise ValueError("Number of features {} implies there is not dimensionality reduction that is possible".format(n_features))

        if self.n_components > n_features:
            logger.warning('Number of components {} cannot be higher than number of features {}'.format(self.n_components, n_features))
            logger.warning('n_components will be set instead to: {}'.format(n_features - 1))
            self.n_components = n_features - 1

        # Check if input matrix is sparse
        # scipy.sparse defines a number of optimized sparse objects and issparse
        # determines if the insput is ot type scipy.sparse matrix object
        # To ntoe some matrixes can still be sparsed but not of that optimized object type
        if sp.issparse(X): # compressed format of type scipy.sparse
            self.sp_issparse = True
            self.issparse = True
            self.sparsity = 1.0 - csr_matrix.getnnz(X) / (X.shape[0] * X.shape[1])
            logger.info('X is sparse and of type scipy.isparse')

        else: # non compressed
            self.sparsity = 1.0 - count_nonzero(X) / X.size
            if self.sparsity > SPARSITY:
                self.issparse = True

        if self.issparse: logger.info('X has a sparsity of: {}'.format(self.sparsity))
        else: logger.info('X is not sparse')

        # Center X
        return DimRed._center(X), n_samples, n_features


    def _eigen_sorted(X_cov):
        """
        Compute the eigen values and vectors using numpy
            and return the eigenvalue and eigenvectors
            sorted based on eigenvalue from high to low
        """
        # Compute the eigen values and vectors using numpy
        eigen_vals, eigen_vecs = np.linalg.eig(X_cov)

        # Sort the eigenvalue and eigenvector from high to low
        idx = eigen_vals.argsort()[::-1]

        return eigen_vals[idx], eigen_vecs[:, idx]


    def _postprocess_sklearn_pca(self, X, pca):
        """
        Postprocessing for sklearn PCA

        Attributes
        components_ : ndarray of shape (n_components, n_features)
            Principal axes in feature space, representing the directions of
            maximum variance in the data. The components are sorted by
            ``explained_variance_``.
        explained_variance_ : ndarray of shape (n_components,)
            The amount of variance explained by each of the selected components.
            Equal to n_components largest eigenvalues
            of the covariance matrix of X.
        explained_variance_ratio_ : ndarray of shape (n_components,)
            Percentage of variance explained by each of the selected components.
            If ``n_components`` is not set then all components are stored and the
            sum of the ratios is equal to 1.0.
        singular_values_ : ndarray of shape (n_components,)
            The singular values corresponding to each of the selected components.
            The singular values are equal to the 2-norms of the ``n_components``
            variables in the lower-dimensional space.
        mean_ : ndarray of shape (n_features,)
            Per-feature empirical mean, estimated from the training set.
            Equal to `X.mean(axis=0)`.
        n_components_ : int
            The estimated number of components. When n_components is set
            to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
            number is estimated from input data. Otherwise it equals the parameter
            n_components, or the lesser value of n_features and n_samples
            if n_components is None.
        n_features_ : int
            Number of features in the training data.
        n_samples_ : int
            Number of samples in the training data.
        noise_variance_ : float
            The estimated noise covariance following the Probabilistic PCA model
            from Tipping and Bishop 1999. See "Pattern Recognition and
            Machine Learning" by C. Bishop, 12.2.1 p. 574 or
            http://www.miketipping.com/papers/met-mppca.pdf. It is required to
            compute the estimated data covariance and score samples.
            Equal to the average of (min(n_features, n_samples) - n_components)
            smallest eigenvalues of the covariance matrix of X.
        """
        self.explained_variance_ = pca.explained_variance_
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.singular_values_ = pca.singular_values_
        self.mean_ = pca.mean_
        self.components_ = pca.components_
        self.n_components_ = pca.n_components_
        self.noise_variance_ = pca.noise_variance_

        logger.info('n_features_: {}'.format(self.n_features_))
        logger.info('n_samples_: {}'.format(self.n_samples_))
        #logger.info('components_: \n{}'.format(self.components_))
        logger.info('n_components_: {}'.format(self.n_components_))
        logger.info('explained_variance_: \n{}'.format(self.explained_variance_))
        logger.info('explained_variance_ratio_: \n{}'.format(self.explained_variance_ratio_))
        logger.info('singular_values_: \n{}'.format(self.singular_values_))
        logger.info('noise_variance_: {}'.format(self.noise_variance_))

        return X


    def _postprocess_sklearn_truncated_svd(self, X, pca):
        """
        Postprocessing for sklearn Truncated SVD

        Attributes:
        components_ : ndarray of shape (n_components, n_features)
        explained_variance_ : ndarray of shape (n_components,)
            The variance of the training samples transformed by a projection to
            each component.
        explained_variance_ratio_ : ndarray of shape (n_components,)
            Percentage of variance explained by each of the selected components.
        singular_values_ : ndarray od shape (n_components,)
            The singular values corresponding to each of the selected components.
            The singular values are equal to the 2-norms of the ``n_components``
            variables in the lower-dimensional space.

        """
        self.components_ = pca.components_
        self.explained_variance_ = pca.explained_variance_
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.singular_values_ = pca.singular_values_
        # Truncated SVD does not have n_components and noise variance
        #self.n_components_ = pca.n_components_
        #self.noise_variance_ = pca.noise_variance_

        logger.info('n_features_: {}'.format(self.n_features_))
        logger.info('n_samples_: {}'.format(self.n_samples_))
        #logger.info('components_: \n{}'.format(self.components_))
        #logger.info('n_components_: {}'.format(self.n_components_))
        logger.info('explained_variance_: \n{}'.format(self.explained_variance_))
        logger.info('explained_variance_ratio_: \n{}'.format(self.explained_variance_ratio_))
        logger.info('singular_values_: \n{}'.format(self.singular_values_))
        #logger.info('noise_variance_: {}'.format(self.noise_variance_))

        return X


    def _postprocess_sklearn_sparsepca(self, X, pca):
        """
        Postprocessing for sklearn SparsePCA

        Attributes
        components_ : ndarray of shape (n_components, n_features)
            Sparse components extracted from the data.
        error_ : ndarray
            Vector of errors at each iteration.
        n_components_ : int
            Estimated number of components.
            .. versionadded:: 0.23
        n_iter_ : int
            Number of iterations run.
        mean_ : ndarray of shape (n_features,)
            Per-feature empirical mean, estimated from the training set.
            Equal to ``X.mean(axis=0)``.
        """
        self.components_ = pca.components_
        self.n_components_ = pca.n_components_
        self.mean_ = pca.mean_

        logger.info('n_features_: {}'.format(self.n_features_))
        logger.info('n_samples_: {}'.format(self.n_samples_))
        #logger.info('components_: \n{}'.format(self.components_))
        logger.info('n_components_: {}'.format(self.n_components_))
        logger.info('mean_: {}'.format(self.mean_))

        return X

    def _postprocess_dimred_pca_svd(self, U, Sigma, Vt):
        """
        Postprocessing for PCA SVD
        """
        n_samples, n_features = U.shape

        if self.n_components is None:
            self.n_components = n_features - 1

        components_ = Vt

        # Get variance explained by singular values
        explained_variance_ = (Sigma ** 2) / (n_samples - 1)

        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = Sigma.copy()  # Store the singular values.

        n_components = self.n_components
        # converting n_components ratio to an integer based on variance
        if 0 < n_components < 1.0:
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, self.n_components,
                                           side='right') + 1

        # Compute noise covariance using Probabilistic PCA model
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.
        self.components_ = components_[0:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        # Project the data
        X_transf = np.empty([n_samples, self.n_components_])
        X_transf[:] = U[:, :self.n_components_]
        X_transf *= Sigma[:self.n_components_]

        #logger.info('n_features_: {}'.format(self.n_features_))
        #logger.info('n_samples_: {}'.format(self.n_samples_))
        #logger.info('components_: \n{}'.format(self.components_))
        logger.info('n_components_: {}'.format(self.n_components_))
        logger.info('explained_variance_: \n{}'.format(self.explained_variance_))
        logger.info('explained_variance_ratio_: \n{}'.format(self.explained_variance_ratio_))
        logger.info('noise_variance_: {}'.format(self.noise_variance_))

        return X_transf


    def _postprocess_dimred_pca_evd(self, X_centered, eigen_vals_sorted, eigen_vecs_sorted):
        """
        Postprocessing for PCA EVD
        """
        n_samples, n_features = X_centered.shape

        # Calculating the explained variance on each of components
        explained_variance_ = np.empty([n_features], dtype=float)
        for i in eigen_vals_sorted:
             np.append(explained_variance_, (i/sum(eigen_vals_sorted))*100)

        # Identifying components that explain at least 95%
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var

        n_components = self.n_components
        # converting n_components ratio to an integer based on variance
        if 0 < n_components < 1.0:
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, self.n_components,
                                           side='right') + 1

        self.components_ = eigen_vecs_sorted[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        #self.noise_variance_ = explained_variance_[n_components:].mean()

        logger.info('n_features_: {}'.format(self.n_features_))
        logger.info('n_samples_: {}'.format(self.n_samples_))
        #logger.info('components_: \n{}'.format(self.components_))
        logger.info('n_components_: {}'.format(self.n_components_))
        logger.info('explained_variance_: \n{}'.format(self.explained_variance_))
        logger.info('explained_variance_ratio_: \n{}'.format(self.explained_variance_ratio_))
        #logger.info('noise_variance_: {}'.format(self.noise_variance_))

        # Project the data
        X_transf = np.empty([n_samples, self.n_components_])
        X_proj = np.dot(X_centered, eigen_vecs_sorted)   # now of shape (n_samples, n_components)
        X_transf[:] = X_proj[:, :self.n_components_]

        return X_transf
