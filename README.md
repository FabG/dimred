# DimRed - Dimension Reduction Package

This python package aims at offering Dimension Reduction capabilities leveraging algorithms such as Principal Component Analysis (PCA) and others such as SVD (TruncatedSVD) and SparsePCA.
Its goal is to reduce the number of features whilst keeping most of the original information.

Why is Dimension Reduction useful?
- Reduces training time — due to smaller dataset
- Removes noise — by keeping only what’s relevant
- Makes visualization possible — in cases where you have a maximum of 3 principal components


This package implements the below existing sklearn packages, and automatically picks the most appropriate based on the data:
 - `sklearn.decomposition.PCA` - if input is not sparse. More info at [link](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
 - `sklearn.decomposition.TruncatedSVD` - if input is sparse. More info at [link](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD)
 - ...


# Installation
You need to run Python 3.X.
And you should set up a virtual environment with `conda` or `virtualenv`

Run:
```
> pip install -r requirements
```

Finally, don't forget to set you `$PYTHONPATH` variable to the root of your projects if you want to run the tests.
```
> export PYTHONPATH=/to/root/of/your/project
```
It should map to: `/your/path/dimred/dimred`

# Tests
For Unit Tests, run:  
`> pytest`
Don't forget to set your `$PYTHONPATH` to the root of your project

If you also want to see the print output to stdout, run:  
`> pytest --capture=tee-sys`

For Unit Tests Coverage, run:  
`> pytest --cov=dimred tests/`

We should aime at having a minimum of 80% code coverage, and preferably closer or equal to 100%.


# Examples


# Notebooks
 - [PCA implementation with EVD and SVD](notebooks/pca_evd_svd.ipynb) => provides implementation of PCA with EVD and SVD and shows SVD is a better implementation


# More information about the algorithm and their parameters

## PCA
When using `PCA` (Principal Component Analysis), you are using a Linear Dimension reduction algorithm, that will project your data to a lower dimensional space. It is a technique for `feature extraction` by combining input variables in a specific way so that the output "new" variables (or components) are all `independant of one another`. This is a benefit because of the assumptions of a linear model.

Notes: PCA is an analysis approach. You can do PCA using SVD, or you can do PCA doing the eigen-decomposition, or you can do PCA using many other methods.  In fact, most implementations of PCA actually use performs SVD under the hood rather than doing eigen decomposition on the covariance matrix because SVD can be much more efficient and is able to handle sparse matrices. In addition, there are reduced forms of SVD which are even more economic to compute.

From a high-level view PCA using the eigen-decomposition has three main steps:
(1) Compute the covariance matrix of the data
(2) Compute the eigen values and vectors of this covariance matrix
(3) Use the eigen values and vectors to select only the most important feature vectors and then transform your data onto those vectors for reduced dimensionality!

### Principal Components


# Resources
 - [scikit learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
 - [MIT open source pca packate](https://github.com/erdogant/pca)
 - [iris dataset for Unit Test](https://archive.ics.uci.edu/ml/datasets/Iris)
 - [mnist handwritten digits dataset for Unit Test](http://yann.lecun.com/exdb/mnist/) - We use for UnitTesting a modified version of MNIST dataset that contains 2000 labeled images of each digit 0 and 1. Images are 28x28 pixels
 - [pca and svd explained with numpy](https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8)
