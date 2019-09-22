import numpy as np
import scipy.linalg

def adjacency_matrix(n, cycles):
    A = np.zeros([n, n])
    for cycle in cycles:
        A[cycle[:, 0], cycle[:, 1]] += 1
        A[cycle[1:, 0], cycle[0:(len(cycle)-1), 1]] += 1
        A[cycle[0, 0], cycle[-1, 1]] += 1

    # check that it is 2-regular
    if not all(np.sum(A, axis=0) == 2) or not all(np.sum(A, axis=1) == 2):
        raise RuntimeError("Not valid cycle decomposition:\n{}\n Gives adjacency matrix:\n {}".format(cycles, A))
    return A

def second_singular_value(A):
    # return np.sort(np.abs(scipy.linalg.eigvals(A)))[-2]
    return scipy.linalg.svd(A)[1][1]

def partition_ssv(p):
    n = np.sum(p)
    cycles = []
    m = 0
    for k in p:
        cycles.append(np.array([[m+x, m+x] for x in range(k)]))
        m += k
    A = adjacency_matrix(n, cycles)
    return second_singular_value(A)

cycles = [
    np.array([[0, 0], [2, 1]]),
    np.array([[1, 2], [3, 3]]),
]
A = adjacency_matrix(4, cycles)
print(second_singular_value(A))

cycles = [
    np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
]
A = adjacency_matrix(4, cycles)
print(second_singular_value(A))
print(np.linalg.svd(A)[1])

cycles = [
    np.array([[0, 0], [1, 1], [2, 3], [3, 2]]),
]
A = adjacency_matrix(4, cycles)
print(second_singular_value(A))
print(np.linalg.svd(A)[1])


cycles = [
    np.array([[0, 0], [1, 2], [2, 3]]),
    np.array([[3, 0]])
]
A = adjacency_matrix(4, cycles)
print(second_singular_value(A))
print(np.linalg.svd(A)[1])

from sympy import Matrix



cycles = [
    np.array([[0, 0], [1, 1], [2, 2]]),
]
A = adjacency_matrix(3, cycles)
print(A)
print(second_singular_value(A))

print("Size 3")
cycles = [
    np.array([[0, 0], [1, 1], [2, 2]]),
]
A = adjacency_matrix(3, cycles)
print(second_singular_value(A))

cycles = [
    np.array([[0, 0], [1, 2]]),
    np.array([[2, 1]]),
]
A = adjacency_matrix(3, cycles)
print(second_singular_value(A))

cycles = [
    np.array([[0, 0], [1, 2], [2, 1]]),
]
A = adjacency_matrix(3, cycles)
print(second_singular_value(A))
# print(A)
# print(np.linalg.svd(A)[1])

cycles = [
    np.array([[0, 1], [1, 2]]),
    np.array([[2, 0]]),
]
A = adjacency_matrix(3, cycles)
print(second_singular_value(A))

n = 16
cycles = [
    np.array([[x, x] for x in range(n)])
]
A = adjacency_matrix(n, cycles)
print(second_singular_value(A))
bound = np.sqrt(2 * (n-2) / (n-1))
print(bound)
print(np.linalg.svd(A)[1])

print(partition_ssv([4, 4, 4, 4]))
