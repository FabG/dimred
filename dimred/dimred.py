"""
dimred.py

DimRed is a python package to perform Dimension Reduction using PCA by default and other algorithms.

"""
from sklearn.decomposition import PCA
import scipy.sparse as sp

class dimred():
    """
    DimRed module
    """

    def __init__(self, algo='pca', n_components=None):
        """
        Initialize dimred with user-defined parameters, defaulting to PCA algorithm

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
        model, explained_variance_ratio = self._fit(X)
        return (model, explained_variance_ratio)

    def _fit(self, X):
        """Dispatch to the right submethod depending on the chosen solver."""

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if sp.issparse(X):
            raise TypeError('PCA does not support sparse input. See '
                            'TruncatedSVD for a possible alternative.')

        if self.algo == 'pca':
            pca = PCA(n_components=2)

            model = PCA(n_components=self.n_components)
            model.fit(X)
            explained_variance_ratio = model.explained_variance_ratio_
            percent_explained_variance = explained_variance_ratio.cumsum()


        return(model, explained_variance_ratio)
