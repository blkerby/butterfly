import numpy as np
from matplotlib import pyplot as plt



class Spline:
    def __init__(self, family, coefs):
        """
        Construct a spline from a given family with given B-spline coefficients.
        :param family (SplineFamily):
        :param coefs (ndarray): 1D-array of B-spline coefficients
        """
        self.family = family
        self.coefs = coefs

    def eval(self, x):
        """
        Evaluate a spline at a given array of points, using de Boor's algorithm
        (based on code sample at https://en.wikipedia.org/wiki/De_Boor%27s_algorithm)
        :param x (ndarray): 1D-array of points at which to evaluate the spline
        :return (ndarray): 1D-array
        """
        degree = self.family.degree
        p = degree
        t = self.family.padded_knots
        c = self.coefs
        c = np.concatenate([np.repeat([c[0]], p), c, np.repeat([c[-1]], p)])
        k = np.clip(np.searchsorted(t, x) - 1, p, len(t) - p - 2)
        x = np.clip(x, t[0], t[-1])
        d = [c[j + k - p] for j in range(0, p + 1)]
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
                alpha = np.where(np.isnan(alpha), 0.0, alpha)
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

        return d[p]

    def derivative(self):
        family = SplineFamily(self.family.knots, self.family.degree - 1, boundary_constraint='zero')




class SplineFamily:
    def __init__(self, knots, degree):
        """
        Construct a family of splines with given `knots` and `degree`, representing the set of
        piecewise-polynomial functions on the real line (with changes in polynomials only at `knots`) of
        degree `degree` and having continuous derivatives up to order `degree - 1`, and subject to the boundary
        conditions of being constant before the first knot and after the last knot.
        :param knots (List[float]): List of knot locations (assumed to be sorted in increasing order)
        :param degree (int): Degree (e.g., 3 for cubic splines, 0 for piecewise constant functions)
        """
        self.knots = knots
        self.degree = degree
        self.dimension = len(self.knots) - degree + 1
        if self.dimension < 0:
            raise ValueError("degree too large for given number of knots")
        self.padded_knots = np.concatenate([np.repeat([knots[0]], degree + 1), knots,
                                            np.repeat([knots[-1]], degree + 1)])

    def bspline(self, coefs):
        return Spline(self, coefs)

family = SplineFamily(np.linspace(-1, 1, 6), 1)
coef = np.zeros(family.dimension)
coef[-1] = 1.0
s = family.bspline(coef)
x = np.linspace(-1.5, 1.5, 1000)
y = s.eval(x)
plt.plot(x, y)
plt.show()
