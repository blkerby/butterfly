import numpy as np
from matplotlib import pyplot as plt
import torch
from numpy.polynomial.legendre import leggauss

class Spline:
    def __init__(self, family, coefs):
        """
        Construct a spline from a given family with given B-spline coefficients.
        :param family (SplineFamily):
        :param coefs (torch.Tensor): 1D-array of B-spline coefficients
        """
        self.family = family
        self.coefs = coefs
        p = self.family.degree
        c0 = coefs[0].repeat([p]) if p > 0 else coefs[0:0]   # Workaround for PyTorch bug/crash in case p=0
        c1 = coefs[-1].repeat([p]) if p > 0 else coefs[0:0]
        self.padded_coefs = torch.cat([c0, coefs, c1])

    def eval(self, x):
        """
        Evaluate a spline at a given array of points, using de Boor's algorithm
        (based on code sample at https://en.wikipedia.org/wiki/De_Boor%27s_algorithm)
        :param x: 1D-array of points at which to evaluate the spline
        :return: 1D-array
        """
        p = self.family.degree
        t = self.family.padded_knots
        c = self.padded_coefs
        k = np.clip(np.searchsorted(t, x) - 1, p, len(t) - p - 2)
        x = torch.clamp(x, t[0], t[-1])
        d = [c[j + k - p] for j in range(0, p + 1)]
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
                alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
        return d[p]

    def derivative(self):
        p = self.family.degree
        family = SplineFamily(self.family.knots, p - 1)
        t = self.family.knots
        c = self.coefs
        deriv_coef = p * (c[1:] - c[:-1]) / (t[p:] - t[:-p])
        return Spline(family, torch.cat([torch.tensor([0.0]), deriv_coef, torch.tensor([0.0])]))

    def inner_product(self, spline):
        """
        Compute the inner product between `self` and `spline` (which assumed to be from the same family of splines),
        namely the integral of the product of the two splines on the interval between the first and last knot.
        :param spline:
        :return: the value of the inner product
        """
        # We use Gauss-Legendre quadrature to compute the integral.
        leg_x = self.family.leg_x
        leg_wt = self.family.leg_wt
        total = 0.0
        knots = self.family.knots
        for i in range(len(knots) - 1):
            t0 = knots[i]
            t1 = knots[i + 1]
            x = leg_x * (t1 - t0) / 2 + (t1 + t0) / 2
            y = self.eval(x) * spline.eval(x)
            total += torch.sum(y * leg_wt * (t1 - t0) / 2)
        return total

    def penalty(self, c):
        """
        Compute the "penalty" of the spline, given penalty coefficients `c` for each order of derivative:
            penalty = sum_{i=0}^{len(c)-1} c_i * integral_{knots[0]}^{knots[-1]} ((d/dx)^i f(x))^2 dx
        :param c: penalty coefficients, i.e., `c[0]` is the penalty coefficient for the 0th-order derivative (i.e.
        the function itself), `c[1]` is the penalty coefficient for the 1st derivative, and so on.
        :return:
        """
        s = self
        out = 0.0
        assert len(c) <= self.family.degree + 1
        for i in range(len(c)):
            out += c[i] * s.inner_product(s)
            if i != len(c) - 1:
                s = s.derivative()
        return out

class SplineFamily:
    def __init__(self, knots, degree):
        """
        Construct the space of splines with given `knots` and `degree`. Specifically, this is an object representing
        the set of piecewise-polynomial functions (with changes in polynomials only at `knots`) of
        polynomial degree `degree` and having continuous derivatives (on the whole real line) up to order `degree - 1`,
        subject to the boundary conditions of being constant before the first knot and after the last knot (i.e., the
        derivatives of order `1, 2, ..., degree - 1` are all zero at the first knot and last knot.
        :param knots (torch.Tensor): 1D array of knot locations (assumed to be sorted in increasing order)
        :param degree (int): Degree (e.g., 3 for cubic splines, 0 for piecewise constant functions)
        """
        self.knots = knots
        self.degree = degree
        self.dimension = len(self.knots) - degree + 1
        if self.dimension < 0:
            raise ValueError("degree too large for given number of knots")
        self.padded_knots = torch.cat([torch.full([degree + 1], knots[0]), knots,
                                       torch.full([degree + 1], knots[-1])])
        leg_x, leg_wt = leggauss(max(1, degree))
        self.leg_x = torch.tensor(leg_x, dtype=knots.dtype, device=knots.device)
        self.leg_wt = torch.tensor(leg_wt, dtype=knots.dtype, device=knots.device)

    def bspline(self, coefs):
        return Spline(self, coefs)

knots = torch.linspace(-1, 1, 6)
degree = 1
family = SplineFamily(knots, degree)
coef = torch.zeros(family.dimension)
coef[0] = 1.0
coef[1] = 1.0
coef[2] = 1.0
coef.requires_grad = True
# s = family.bspline(coef).derivative()
# s = family.bspline(coef)
# print(s.coefs)
# print(s.inner_product(s))
s = family.bspline(coef)
# x = torch.linspace(-1.0, 1.0, 10000)
# y = s.eval(x)

pen = s.penalty([0.0, 1.0])
print(pen)

# y[0].backward()
#
#
# plt.plot(x, y)
# plt.show()
# torch.mean(y**2 * 2)