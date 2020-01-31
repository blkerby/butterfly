import torch
import linalg
import logging

def subproblem_solve(a, d, y, max_iter):
    """Given the function f(x) defined by
        f(x) = sum_i a[i] / (x + d[i])^2
    where a[i] >= 0, d[i] >= 0, a[0] > 0, and y > 0, use Newton's method to return the unique solution to
    f(x) = y with x > 0.
    """
    # logging.info("subproblem_solve: a={}, d={}, y={}".format(a, d, y))
    x0 = torch.max(torch.sqrt(a / y) - d) * (1.0 + linalg.eps(a.dtype))
    for i in range(max_iter):
        f0 = torch.sum(a / (x0 + d)**2)
        df0 = 2*torch.sum(a / (x0 + d)**3)
        x1 = x0 + (f0 - y) / df0
        if x1 <= x0 or df0 == 0.0:
            # Our method ensures that in exact arithmetic the estimates of x would increase monotonically, so if it
            # (weakly) decreases, then we can infer that we have reached numerical convergence up to machine precision.
            return x1
        x0 = x1
    raise RuntimeError("Newton root search failed to converge")

def permute(x, ind):
    out = torch.empty_like(x)
    out[ind] = x
    return out

def trust_region_solve(d, g, r, max_iter=50):
    """Compute a minimum of the function
         f(x) = g^T*x + 1/2 * x^T*D*x
    subject to |x| <= r, and D=diag(d). Except for edge cases,
    the solution will have the form x = -(D + lam*I)^(-1) * g, for some value of scalar lam >= 0.

    Returns: `x` where a minimum occurs.

    For details see Nocedal & Wright's "Numerical Optimization", Chapter 4.
    """

    d, ind = d.sort()
    g = g[ind]

    g2 = g ** 2
    r2 = r ** 2
    if d[0] > 0 and torch.sum(g2 / d**2) <= r2:
        # The Hessian is positive-definite and the Newton step is within the trust region, so the Newton step is the solution.
        return permute(-g / d, ind)
    else:
        # We reparametrize `d` for better numerical stability. It can happen that the solution value for `lam` is only
        # larger than `-d[0]` by a very small amount, less than the floating point epsilon; by reparametrizing we
        # prevent underflow or loss of precision in this case. So the value `l0` that we compute below would
        # correspond to the solution lam = d[0] + l0, although we never explicitly compute this sum.
        d0 = d - d[0]

        # Determine the multiplicity of the smallest eigenvalue (corresponding to directions of most negative curvature):
        mult = 1
        for i in range(1, d.shape[0]):
            if d0[i] != 0:
                break
            mult += 1
        if torch.all(g2[:mult] == 0):
            # We are in the "hard case", where the gradient is orthogonal to all the directions of most negative curvature.
            f0 = torch.sum(g2[mult:] / d0[mult:]**2)
            if f0 > r2:
                l0 = subproblem_solve(g2[mult:], d0[mult:], r2, max_iter)
            else:
                # This is the edge case where the solution is non-unique: it involves making an arbitrary choice of a
                # direction of most negative curvature (we select the first standard basis vector):
                w = torch.sqrt(r2 - f0)
                z = torch.zeros_like(g)
                z[mult:] = -g[mult:] / d0[mult:]
                z[0] = w
                return permute(z, ind)
        else:
            # Normal case where solution is on the boundary of the trust region: we perform Newton's root finding
            # (in one dimension) to determine the appropriate value for `lam`
            l0 = subproblem_solve(g2, d0, r2, max_iter)
        return permute(-g / (l0 + d0), ind)
