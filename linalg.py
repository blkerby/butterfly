import torch

def stable_norm(u):
    """Compute the Euclidean norm of a vector `u` in a numerically stable way. We want a non-zero vector to always have
    a non-zero norm, which, due to the possibility of underflow, is not ensured by the PyTorch function torch.norm() on
    its own (at least not for the torch.float64 datatype; for torch.float32 this might already be the case, at least in
    one configuration, but this is undocumented and we do not want to rely on it.)"""
    m = u.abs().max()
    if m == 0.0:
        return m
    u = u / m
    return u.norm(2) * m


def eps(dtype):
    if dtype == torch.float64:
        return 1e-15
    elif dtype == torch.float32:
        return 1e-6
    else:
        raise RuntimeError("Unexpected dtype: {}".format(dtype))


def expand_Q(Q, u, k, max_iter=3):
    """Given a matrix `Q`, considered as only defined on its first k columns (the remaining columns being assumed to
    be unallocated memory), which are assumed to be orthonormal, modify Q by possibly adding a new column (in place,
    by replacing its (k+1)st column) with a new unit-length column orthogonal to the first `k` columns, determining a
    vector `r` (of length `k` or `k+1`) such that after this operation `Q*r = u`. The situation where `r` has length `k`
    would be when `u` is already numerically equal to a linear combination of the columns of `Q` (in particular this
    happens if `Q` is square.).
    """
    Q0 = Q[:, :k]
    nm0 = stable_norm(u)

    # Subtract away from `u` its projection onto the columns of `Q`; the new value of `u` will then be approximately
    # orthogonal to the columns of `Q`.
    r = Q0.t().mv(u)
    if Q0.shape[0] == k:
        # `Q` is already square, so we already have the required `r` without needing to add a column to `Q` (and
        # in any case it would be impossible to add a column while retaining orthonormality)
        return r
    u = u - Q0.mv(r)

    nm1 = nm0
    for i in range(max_iter):
        nm2 = stable_norm(u)
        if nm2 >= 0.5 * nm1:
            break
        elif nm2 <= eps(u.dtype) * nm0:
            # The original `u` was numerically already equal to a linear combination of the columns of `Q`, so there is no
            # need to add another column (and adding `u1` as a column would risk destroying the orthonormality of `Q`).
            return r
        # For numerical stability we subtract away the projection of `u` onto the columns `Q` again. This is based
        # on a similar idea to the Modified Gram-Schmidt method, except this way is faster and more accurate. This is
        # important to do, since otherwise in certain cases the orthonormality of `Q` could be completely ruined
        # (e.g., see https://fgiesen.wordpress.com/2013/06/02/modified-gram-schmidt-orthogonalization/)
        r1 = Q0.t().mv(u)
        u -= Q0.mv(r1)
        r += r1
        nm1 = nm2
    else:
        raise RuntimeError("expand_Q failed to converge")

    u /= nm2
    Q[:, k] = u
    r = torch.cat([r, nm2.view(-1)])
    return r

def spectral_update(Q, M, lam0, u, c, k):
    """
    Given
    - a matrix `Q`, considered as only defined on its first k columns (the remaining columns being assumed to
    be unallocated memory), which are assumed to be orthonormal
    - a square matrix `M`, considered as only defined on its first k rows and columns
    - a vector `u` having the same number of entries as `Q` has rows
    - scalars `lam0` and `c`
    determines matrices `Q1` and `M1` such that
        Q1*M1*Q1^T + lam0*(I - Q1*Q1^T) = Q*M*Q^T + lam0*(I - Q*Q^T) + c*u*u^T
    `Q` are `M` are modified in place so that they are replaced with `Q1` and `M1` respectively.
    """
    r = expand_Q(Q, u, k)
    m = r.shape[0]
    if m > k:
        M[k, :] = 0.0
        M[:, k] = 0.0
        M[k, k] = lam0
    M[:m, :m].add_(c, torch.ger(r, r))
    return m
