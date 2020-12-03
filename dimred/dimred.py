"""
dimred.py

DimRed is a python package to perform Dimension Reduction using PCA by default and other algorithms.

"""
from sklearn.decomposition import PCA

class dimred():
    """
    DimRed module
    """

    def __init__(self, algo="pca", n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None):
        """
        Initialize dimred with user-defined parameters, defaulting to PCA algorithm

        Parameters
        ----------
        n_components : Number of components to keep.
            Missing Value => All components are kept.
            Values > 0 are the number of Top components.
                Ex: n_components = 3 => returns Top 3 principal components
            Values < 0 are the components that cover at least the percentage of variance.
                Ex: n_components = 0.85 => returns all components that cover at least 85% of variance.
        """


        # Store in object

        pass
