# DimRed - Dimension Reduction Package


### 1. DimRed Introduction
**DimRed** is a python package that enables **Dimension Reduction** and visualization.
It offers:
 - internally built SVD and EVD algorithms with `numpy`, namely:
  - `dimred_pca_svd` - Dimension reduction using the Singular Value Decomposition: `X . V = U . S ==> X = U.S.Vt`
  - `dimred_pca_evd`- Dimension reduction using the Eigen Value Decomposition
 - support for sklearn `decomposition` algorithms as a "pass-through", namely:
   - `PCA()` - see [sklearn_doc](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) for more info
   - `TruncatedSVD()` for Sparse matrices- see [sklearn_doc](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) for more info
   - `SparsePCA()` for Sparse matrices- see [sklearn_doc](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html) for more info


#### What is Dimension Reduction?
**Dimension reduction** (or Dimensionality reduction) refers to techniques for reducing the number of input variables in training data.

*When dealing with high dimensional data, it is often useful to reduce the dimensionality by projecting the data to a lower dimensional subspace which captures the “essence” of the data. This is called **dimensionality reduction**.*

— Page 11, [Machine Learning: A Probabilistic Perspective](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020/ref=as_li_ss_tl?keywords=Machine+Learning:+A+Probabilistic+Perspective&qid=1580679017&sr=8-1&linkCode=sl1&tag=inspiredalgor-20&linkId=e1ce409a189df7eeb214b15424a7379c&language=en_US), 2012.


High-dimensionality might mean hundreds, thousands, or even millions of input variables.

Fewer input dimensions often means correspondingly fewer parameters or a simpler structure in the machine learning model, referred to as degrees of freedom. A model with too many degrees of freedom is likely to **overfit** the training dataset and may not perform well on new data.

It is **desirable to have simple models that generalize well**, and in turn, input data with few input variables. This is particularly true for linear models where the number of inputs and the degrees of freedom of the model are often closely related.



#### Why is Dimension Reduction useful?
- Reduces training time — due to smaller dataset
- Removes noise — by keeping only what’s relevant
- Makes visualization possible — in cases where you have a maximum of 3 principal components

#### How is DimRed implemented and what capabilities does it offer?
DimRed is offered as a native python package (and soon part of MLaaS as a service). Under the hood, it leverages `numpy` and `scikit-learn` to implement and expose:
 - **PCA** (Principal Component Analysis)
 - **SVD** (Singular Value Decomponsition)
 - **EVD** (Eigen Values Decompoisition)


See section 5(Dimension Reduction - Additional info) below for more information about each algorithm

### 2. DimRed Installation
#### 2.1 Install
You need to run Python 3.X.
And you should set up a virtual environment with `conda` or `virtualenv`

Run:
```bash
> pip install -r requirements
```

Finally, don't forget to set you `$PYTHONPATH` variable to the root of your projects if you want to run the tests.
```bash
> export PYTHONPATH=/to/root/of/your/project
```
It should map to: `/your/path/dimred/dimred`

#### 2.2 Tests
For Unit Tests, run:  
```bash
> pytest
```
Don't forget to set your `$PYTHONPATH` to the root of your project

If you also want to see the print output to stdout, run:  
```bash
> pytest --capture=tee-sys
```

For Unit Tests Coverage, run:  
```bash
> pytest --cov=dimred tests/
```

Or:  
```bash
> pytest --capture=tee-sys --cov=dimred tests/
```

We should aim at having a minimum of 80% code coverage, and preferably closer or equal to 100%.


### 3. DimRed Examples


### 4. Dimension Reduction Notebooks
 - [PCA implementation with EVD and SVD](notebooks/pca_evd_svd.ipynb) => provides implementation of PCA with EVD and SVD and shows SVD is a better implementation
 - [PCA vs LDA and PCA visualization on Iris data](notebooks/pca_lda_iris.ipynb)

### 5. Dimension Reduction - Additional info

#### 5.1 Dimensionality Reduction Algorithms

There are many algorithms that can be used for dimensionality reduction.

Two main classes of methods are those drawn from linear algebra and those drawn from manifold learning:

##### => Linear Algebra Methods
Matrix factorization methods drawn from the field of linear algebra can be used for dimensionality.
Some of the more popular methods include:
- [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis#:~:text=Principal%20component%20analysis%20(PCA)%20is,components%20and%20ignoring%20the%20rest.): Principal Components Analysis => process of computing the principal components and using them to perform a change of basis on the data, sometimes using only the first few principal components and ignoring the rest.
- [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition): Singular Value Decomposition
- [NMF](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization): Non-Negative Matrix Factorization

For more on matrix factorization, see this [tutorial](https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/)



##### => Manifold Learning Methods
Manifold learning methods seek a lower-dimensional projection of high dimensional input that captures the salient properties of the input data.

Some of the more popular methods include:
- [Isomap](https://en.wikipedia.org/wiki/Isomap) Embedding
- [LLE](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Locally-linear_embedding): Locally Linear Embedding
- [MDS](https://en.wikipedia.org/wiki/Multidimensional_scaling): Multidimensional Scaling
- [Spectral Embedding](https://en.wikipedia.org/wiki/Spectral_clustering)
- [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding): t-distributed Stochastic Neighbor Embedding

Each algorithm offers a different approach to the challenge of discovering natural relationships in data at lower dimensions.

There is no best dimensionality reduction algorithm, and no easy way to find the best algorithm for your data without using controlled experiments.



#### 5.2 DimRed Package - Supported Algorithms

#### PCA
When using `PCA` (Principal Component Analysis), you are using a Linear Dimension reduction algorithm, that will project your data to a lower dimensional space. It is an **unsupervised** technique for `feature extraction` by combining input variables in a specific way so that the output "new" variables (or components) are all `independant of one another`. This is a benefit because of the assumptions of a linear model.

PCA aims to find linearly uncorrelated orthogonal axes, which are also known as principal components (PCs) in the m dimensional space to project the data points onto those PCs. The first PC captures the largest variance in the data. Let’s intuitively understand PCA by fitting it on a 2-D data matrix, which can be conveniently represented by a 2-D scatter plot:
<p align="center" width="100%">
    <img width="70%" src="images/pca_animation.gif">
    <br><i>Making sense of PCA by fitting on a 2-D dataset<a href="https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues/140579#140579"> (source)</a></i>
</p>

Since all the PCs (Principal Components) are orthogonal to each other, we can use a pair of perpendicular lines in the 2-D space as the two PCs. To make the first PC capture the largest variance, we rotate our pair of PCs to make one of them optimally align with the spread of the data points. Next, all the data points can be projected onto the PCs, and their projections (red dots on PC1) are essentially the resultant dimensionality-reduced representation of the dataset. Viola, we just reduced the matrix from 2-D to 1-D while retaining the largest variance!

The PCs can be determined via eigen decomposition of the covariance matrix C. After all, the geometrical meaning of eigen decomposition is to find a new coordinate system of the eigenvectors for C through rotations.
Image for post
Eigendecomposition of the covariance matrix C:  
<p align="center" width="100%">
    <img width="40%" src="images/eidgen_decomposition_covariance.png">
</p>

In the equation above, the covariance matrix C(m×m) is decomposed to a matrix of eigenvectors W(m×m) and a diagonal matrix of m eigenvalues Λ. The eigenvectors, which are the column vectors in W, are in fact the PCs we are seeking. We can then use matrix multiplication to project the data onto the PC space. For the purpose of dimensionality reduction, we can project the data points onto the first k PCs as the representation of the data:  
<p align="center" width="100%">
    <img width="30%" src="images/projected_data.png">
</p>

Notes: PCA is an analysis approach. You can do PCA using SVD, or you can do PCA doing the eigen-decomposition, or you can do PCA using many other methods.  In fact, most implementations of PCA actually use performs SVD under the hood rather than doing eigen decomposition on the covariance matrix because SVD can be much more efficient and is able to handle sparse matrices. In addition, there are reduced forms of SVD which are even more economic to compute.

From a high-level view PCA using EVD (eigen-decomposition) has three main steps:
- (1) Compute the covariance matrix of the data
- (2) Compute the eigen values and vectors of this covariance matrix
- (3) Use the eigen values and vectors to select only the most important feature vectors and then transform your data onto those vectors for reduced dimensionality!

PCA can be very easily implemented with numpy as the key function performing eigen decomposition `np.linalg.eig` is already built-in:
```python
def pca(X):
  # Data matrix X, assumes 0-centered
  n, m = X.shape
  assert np.allclose(X.mean(axis=0), np.zeros(m))
  # Compute covariance matrix
  C = np.dot(X.T, X) / (n-1)
  # Eigen decomposition
  eigen_vals, eigen_vecs = np.linalg.eig(C)
  # Project X onto PC space
  X_pca = np.dot(X, eigen_vecs)
  return X_pca
```


#### LDA
Both LDA and PCA are linear transformation techniques: LDA is a supervised whereas PCA is unsupervised – PCA ignores class labels.

**LDA** is very useful to find dimensions which aim at **separating cluster**, thus you will have to know clusters before. LDA is not necessarily a classifier, but can be used as one. Thus LDA can only be used in **supervised learning**

=> *LDA is used to carve up multidimensional space.*LDA is for classification, it almost always outperforms Logistic Regression when modeling small data with well separated clusters. It also handles multi-class data and class imbalances.


To contrast with LDA, **PCA** is a general approach for **denoising and dimensionality reduction** and does not require any further information such as class labels in supervised learning. Therefore it can be used in **unsupervised learning**.

=> *PCA is used to collapse multidimensional space*. PCA allows the collapsing of hundreds of spatial dimensions into a handful of lower spatial dimensions while preserving 70% - 90% of the important information. 3D objects cast 2D shadows. We can see the shape of an object from it's shadow. But we can't know everything about the shape from a single shadow. By having a small collection of shadows from different (globally optimal) angles, then we can know most things about the shape of an object. PCA helps reduce the 'Curse of Dimensionality' when modeling.


#### EVD and SVD
##### SVD - Singular Value Decomposition

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

SVD is another decomposition method for both real and complex matrices. It decomposes a matrix into the product of two unitary matrices (U, V*) and a rectangular diagonal matrix of singular values (Σ):


Here is a friendler  way to visualize the **SVD** formula:
<p align="center" width="100%">
    <img width="40%" src="images/svd_matrix.png">
    <br><i>Illustration of SVD<a href="https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8"> (source)</a></i>
</p>

In most cases, we work with real matrix X, and the resultant unitary matrices U and V will also be real matrices. Hence, the conjugate transpose of the U is simply the regular transpose.

What **SVD** it used for?

[Wikipedia has a nice list](https://en.wikipedia.org/wiki/Singular-value_decomposition#Applications_of_the_SVD), but I'll list a couple.
- One of the most common applications is obtaining a low-rank approximation to a matrix (see **PCA**), which is used for compression, speed-up, and also actual data analysis.
- The other one is for characterizing the pseudo-inverse for analysis or proofs, since inverting it automatically gives a formula that's the inverse when the matrix is invertible, and the pseudo-inverse when it is not.


SVD has also already been implemented in numpy as `np.linalg.svd`. To use SVD to transform your data:

```python
def svd(X):
  # Data matrix X, X doesn't need to be 0-centered
  n, m = X.shape
  # Compute full SVD
  U, Sigma, Vh = np.linalg.svd(X,
      full_matrices=False, # It's not necessary to compute the full matrix of U or V
      compute_uv=True)
  # Transform X with SVD components
  X_svd = np.dot(U, np.diag(Sigma))
  return X_svd
```

##### Relationship between PCA and SVD
PCA and SVD are closely related approaches and can be both applied to decompose any rectangular matrices. We can look into their relationship by performing SVD on the covariance matrix C:
<p align="center" width="100%">
    <img width="40%" src="images/covariance_matrix_pca_svd.png">
</p>

From the above derivation, we notice that the result is in the same form with eigen decomposition of C, we can easily see the relationship between singular values (Σ) and eigenvalues (Λ):
<p align="center" width="100%">
    <img width="40%" src="images/eigen_singular_values_relationship.png">
</p>


To confirm that with numpy:
```python
# Compute covariance matrix
C = np.dot(X.T, X) / (n-1)
# Eigen decomposition
eigen_vals, eigen_vecs = np.linalg.eig(C)
# SVD
U, Sigma, Vh = np.linalg.svd(X,
    full_matrices=False,
    compute_uv=True)
# Relationship between singular values and eigen values:
print(np.allclose(np.square(Sigma) / (n - 1), eigen_vals)) # True
```

So what does this imply?  
It suggests that we can actually perform PCA using SVD, or vice versa. In fact, **most implementations of PCA actually use performs SVD under the hood** rather than doing eigen decomposition on the covariance matrix because SVD can be much more efficient and is able to handle sparse matrices. In addition, there are reduced forms of SVD which are even more economic to compute.


##### EVD - Eigenvalue (spectral) decomposition
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

using the SVD to perform PCA makes much better sense numerically than forming the covariance matrix (in EVD) to begin with, since the formation of 𝐗𝐗⊤ can cause loss of precision


### 6. Resources
#### Articles
 - [scikit learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
 - [MIT open source pca packate](https://github.com/erdogant/pca)
 - [PCA and SVD explained with numpy](https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8)
 - [Mathematical explanation of PCA and SVD](https://math.stackexchange.com/questions/3869/what-is-the-intuitive-relationship-between-svd-and-pca)
 - [Mathematical explanation - how to use SVD to perform PCA](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca)
 - [Tutorial on Principal Component Analysis - White paper](https://arxiv.org/pdf/1404.1100.pdf)
 - [implement PCA using SVD with sklearn and numpy](https://stackoverflow.com/questions/60508233/python-implement-a-pca-using-svd)
 - [EVD and SVD white paper](https://www.cc.gatech.edu/~dellaert/pubs/svd-note.pdf)
 - [Difference between EVD and SVD](https://math.stackexchange.com/questions/320220/intuitively-what-is-the-difference-between-eigendecomposition-and-singular-valu)
 - [Dimension reduction algorithms in python](https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/)
 - [PCA explained on wine data with chart animation](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues/140579#140579)

#### DataSets (for Unit Tests)
We are using various datasets:
 - some created statically in the unit test functions
 - some loaded from scikitlearn: `from sklearn import datasets
`- some available as flat files under: `/tests/data`
  - [mnist handwritten digits dataset for Unit Test](http://yann.lecun.com/exdb/mnist/)
