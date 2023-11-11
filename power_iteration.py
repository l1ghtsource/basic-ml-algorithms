import numpy as np


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    n = data.shape[0]
    eigenvector = np.random.rand(n)

    for _ in range(num_steps):
        eigenvector = np.dot(data, eigenvector)
        eigenvector /= np.linalg.norm(eigenvector)

    eigenvalue = np.dot(eigenvector, np.dot(data, eigenvector)).item()

    return eigenvalue, eigenvector
