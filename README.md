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


### Installation
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

### Tests
For Unit Tests, run:  
`> pytest`
Don't forget to set your `$PYTHONPATH` to the root of your project

If you also want to see the print output to stdout, run:  
`> pytest --capture=tee-sys`

For Unit Tests Coverage, run:  
`> pytest --cov=dimred tests/`

We should aime at having a minimum of 80% code coverage, and preferably closer or equal to 100%.


### Examples


### Notebooks
 - [PCA implementation with EVD and SVD](notebooks/pca_evd_svd.ipynb) => provides implementation of PCA with EVD and SVD and shows SVD is a better implementation
 - [PCA vs LDA and PCA visualization on Iris data](notebooks/pca_lda_iris.ipynb)

### More information about the algorithms and their parameters


#### EVD and SVD
###### SVD - Singular Value Decomposition

The **SVD** allows to describe the effect of a matrix 𝐴 on a vector (via the matrix-vector product), as a three-step process `𝐴=𝑈Σ𝑉†`:
- 1. A first rotation in the input space (`𝑉`)
- 2. A simple positive scaling that takes a vector in the input space to the output space (`Σ`)
- 3. And another rotation in the output space (`𝑈`)

*Note that `𝑉†` denotes the conjugate of `𝑉⊤`, hence the two are equal when 𝑉 is real.*
Note that the conditions above are mathematically the following constraints:

- `𝑉†𝑉=𝐼`    (i.e. 𝑉 is a rotation matrix)
- `Σ=diag(𝜎⃗ )` and `𝜎⃗ ≥0⃗` (`diag` just returns a diagonal matrix with the given diagonal)
- `𝑈†𝑈=𝐼`    (i.e. 𝑈 is a rotation matrix)

The [fundamental theorem of linear algebra](https://en.wikipedia.org/wiki/Fundamental_theorem_of_linear_algebra) says that such a decomposition always exists.

What **SVD** it used for?

[Wikipedia has a nice list](https://en.wikipedia.org/wiki/Singular-value_decomposition#Applications_of_the_SVD), but I'll list a couple.
- One of the most common applications is obtaining a low-rank approximation to a matrix (see **PCA**), which is used for compression, speed-up, and also actual data analysis.
- The other one is for characterizing the pseudo-inverse for analysis or proofs, since inverting it automatically gives a formula that's the inverse when the matrix is invertible, and the pseudo-inverse when it is not.


###### EVD - Eigenvalue (spectral) decomposition
Similarly, for the **eigendecomposition** (also known as eigenvalue decomposition, spectral decomposition, or diagonalization):

An eigendecomposition describes the effect of a matrix 𝐴 on a vector as a different 3-step process `𝐴=𝑄Λ𝑄−1`:
- 1. An invertible linear transformation `(𝑄−1)`
- 2. A scaling `(Λ)`
- 3. The inverse of the initial transformation `(𝑄)`

Correspondingly, these conditions imply the following constraints:
- `𝑄` is invertible
- `Λ=diag(𝜆⃗ )`

This decomposition doesn't always exist, but the [spectral theorem](https://en.wikipedia.org/wiki/Spectral_theorem) describes the conditions under which such a decomposition exists.

Note the most basic requirement is that `𝐴` be a **square matrix** (but this is not enough).

What **EVD** is used for?

- It gives you the ability to efficiently raise a matrix to a large power: 𝐴𝑛=𝑄Λ𝑛𝑄−1. For this reason (and others) it's used heavily in engineering to, say, efficiently analyze and predict the behavior of a linear dynamical system at a future point in time, since this is much faster than manually exponentiating the matrix directly.
- It's also used to analyze the response of a linear system at different frequencies. (Sinusoids of different frequencies are orthogonal, so you get the orthogonal diagonalizability for free.)
- Furthermore, it's also a problem that repeatedly comes up when solving differential equations analytically.


#### EVD Vs SVD
Consider the eigendecomposition `𝐴=𝑃𝐷𝑃−1` and SVD `𝐴=𝑈Σ𝑉∗`. Some key differences are as follows,

- The vectors in the eigendecomposition matrix 𝑃 are not necessarily orthogonal, so the change of basis isn't a simple rotation. On the other hand, the vectors in the matrices 𝑈 and 𝑉 in the SVD are orthonormal, so they do represent rotations (and possibly flips).
- In the SVD, the nondiagonal matrices 𝑈 and 𝑉 are not necessairily the inverse of one another. They are usually not related to each other at all. In the eigendecomposition the nondiagonal matrices 𝑃 and 𝑃−1 are inverses of each other.
- In the SVD the entries in the diagonal matrix Σ are all real and nonnegative. In the eigendecomposition, the entries of 𝐷 can be any complex number - negative, positive, imaginary, whatever.
- The SVD always exists for any sort of rectangular or square matrix, whereas the eigendecomposition can only exists for square matrices, and even among square matrices sometimes it doesn't exist.



#### PCA
When using `PCA` (Principal Component Analysis), you are using a Linear Dimension reduction algorithm, that will project your data to a lower dimensional space. It is an **unsupervised** technique for `feature extraction` by combining input variables in a specific way so that the output "new" variables (or components) are all `independant of one another`. This is a benefit because of the assumptions of a linear model.

Notes: PCA is an analysis approach. You can do PCA using SVD, or you can do PCA doing the eigen-decomposition, or you can do PCA using many other methods.  In fact, most implementations of PCA actually use performs SVD under the hood rather than doing eigen decomposition on the covariance matrix because SVD can be much more efficient and is able to handle sparse matrices. In addition, there are reduced forms of SVD which are even more economic to compute.

From a high-level view PCA using the eigen-decomposition has three main steps:
- (1) Compute the covariance matrix of the data
- (2) Compute the eigen values and vectors of this covariance matrix
- (3) Use the eigen values and vectors to select only the most important feature vectors and then transform your data onto those vectors for reduced dimensionality!



#### LDA
Both LDA and PCA are linear transformation techniques: LDA is a supervised whereas PCA is unsupervised – PCA ignores class labels.

**LDA** is very useful to find dimensions which aim at **separating cluster**, thus you will have to know clusters before. LDA is not necessarily a classifier, but can be used as one. Thus LDA can only be used in **supervised learning**

=> *LDA is used to carve up multidimensional space.*LDA is for classification, it almost always outperforms Logistic Regression when modeling small data with well separated clusters. It also handles multi-class data and class imbalances.


To contrast with LDA, **PCA** is a general approach for **denoising and dimensionality reduction** and does not require any further information such as class labels in supervised learning. Therefore it can be used in **unsupervised learning**.

=> *PCA is used to collapse multidimensional space*. PCA allows the collapsing of hundreds of spatial dimensions into a handful of lower spatial dimensions while preserving 70% - 90% of the important information. 3D objects cast 2D shadows. We can see the shape of an object from it's shadow. But we can't know everything about the shape from a single shadow. By having a small collection of shadows from different (globally optimal) angles, then we can know most things about the shape of an object. PCA helps reduce the 'Curse of Dimensionality' when modeling.



### Resources
#### Articles
 - [scikit learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
 - [MIT open source pca packate](https://github.com/erdogant/pca)
 - [pca and svd explained with numpy](https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8)
 - [Mathematical explanation of PCA and SVD](https://math.stackexchange.com/questions/3869/what-is-the-intuitive-relationship-between-svd-and-pca)
 - [Mathematical explanation - how to use SVD to perform PCA](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca)
 - [Tutorial on Principal Component Analysis - White paper](https://arxiv.org/pdf/1404.1100.pdf)
 - [implement PCA using SVD with sklearn and numpy](https://stackoverflow.com/questions/60508233/python-implement-a-pca-using-svd)
 - [EVD and SVD white paper](https://www.cc.gatech.edu/~dellaert/pubs/svd-note.pdf)
 - [Difference between EVD and SVD](https://math.stackexchange.com/questions/320220/intuitively-what-is-the-difference-between-eigendecomposition-and-singular-valu)

#### DataSets (for Unit Test)
They are available under: `/tests/data`
- [iris dataset for Unit Test](https://archive.ics.uci.edu/ml/datasets/Iris)
- [mnist handwritten digits dataset for Unit Test](http://yann.lecun.com/exdb/mnist/)
