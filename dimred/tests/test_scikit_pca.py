import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)

explained_variance_ratio = pca.explained_variance_ratio_
singular_values = pca.singular_values_

print(explained_variance_ratio)
print(singular_values)

assert(explained_variance_ratio[0] == 0.9924428900898052)
assert(explained_variance_ratio[1] == 0.007557109910194766)

assert(singular_values[0] == 6.300612319734663)
assert(singular_values[1] == 0.5498039617971033)
