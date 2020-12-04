# DimRed - Dimension Reduction Package

This python package aims at offering Dimension Reduction capabilities leveraging algorithms such as Principal Component Analysis (PCA) and others.
The core of PCA is built on scikitlearn functionality.
Besides PCA, this package will be extended to offer additional capabilities such as SVD (TruncatedSVD) and SparsePCA.
Ultimately, this package will also offer some "auto-ml" like capabilities picking the best approach based on your data.

# Installation
You need to run Python 3.X.
And you should set up a virtual environment with `conda` or `virtualenv`

Run:
`pip install -r requirements`

Finally, don't forget to set you `$PYTHONPATH` variable to the root of your projects if you want to run the tests.
`export PYTHONPATH=/to/root/of/your/project`

# Tests
Run:
`pytest`
Don't forget to set your `$PYTHONPATH` to the root of your project

If you also want to see the print output to stdout, run:
`pytest --capture=tee-sys`

# Examples


# Resources
 - [scikit learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
 - [MIT open source pca packate](https://github.com/erdogant/pca)
 - [iris dataset for Unit Test](https://archive.ics.uci.edu/ml/datasets/Iris)
 - [mnist handwritten digits dataset for Unit Test](http://yann.lecun.com/exdb/mnist/) - We use for UnitTesting a modified version of MNIST dataset that contains 2000 labeled images of each digit 0 and 1. Images are 28x28 pixels
