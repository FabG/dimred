import numpy as np
from dimred import dimred

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = dimred(n_components=2)
model, explained_variance_ratio = pca.fit(X)

print(explained_variance_ratio)

assert(explained_variance_ratio[0] == 0.9924428900898052)
assert(explained_variance_ratio[1] == 0.007557109910194766)
