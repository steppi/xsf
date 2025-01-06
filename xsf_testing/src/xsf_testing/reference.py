import math
import numpy as np
import scipy
import scipy.special as special
import sys

from mpmath import mp  # type: ignore
from mpmath.calculus.optimization import Bisection, Secant
from packaging import version
from typing import overload, Tuple

from xsf_testing.util import reference_implementation


if version.parse(scipy.__version__) >= version.parse("1.16"):
    raise RuntimeError(
        f"SciPy {scipy.__version__} is not an independent reference. SciPy"
        " depends on xsf as of version 1.16."
        )


@overload
def airy(x: float) -> Tuple[float, float, float, float]: ...
@overload
def airy(x: complex) -> Tuple[complex, complex, complex, complex]: ...


@reference_implementation()
def airy(z):
    """Airy functions and their derivatives.

    Notes
    -----
    Airy functions are entire
    """
    ai = mp.airyai(z)
    aip = mp.airyai(z, derivative=1)
    bi = mp.airybi(z)
    bip = mp.airybi(z, derivative=1)
    return ai, aip, bi, bip


@overload
def airye(x: float) -> Tuple[float, float, float, float]: ...
@overload
def airye(x: complex) -> Tuple[complex, complex, complex, complex]: ...


@reference_implementation()
def airye(z):
    """Exponentially scaled Airy functions and their derivatives.

    Notes
    -----
    Scaled Airy functions are entire
    """
    eai = mp.airyai(z) * mp.exp(mp.mpf("2.0") / mp.mpf("3.0") * z * mp.sqrt(z))
    eaip = mp.airyai(z, derivative=1) * mp.exp(
        mp.mpf("2.0") / mp.mpf("3.0") * z * mp.sqrt(z)
    )
    ebi = mp.airybi(z) * mp.exp(
        -abs(mp.mpf("2.0") / mp.mpf("3.0") * (z * mp.sqrt(z)).real)
    )
    ebip = mp.airybi(z, derivative=1) * mp.exp(
        -abs(mp.mpf("2.0") / mp.mpf("3.0") * (z * mp.sqrt(z)).real)
    )
    return eai, eaip, ebi, ebip


@reference_implementation()
def bdtr(k: float, n: float, p: float) -> float:
    """Binomial distribution cumulative distribution function."""
    k, n = mp.floor(k), mp.floor(n)
    with mp.workprec(max(mp.prec, int(mp.ceil(-mp.log(abs(p), b=2))) + 53)):
        # set the precision high enough that mp.one - p != 1
        result = mp.betainc(n - k, k + 1, 0, 1 - p, regularized=True)
    return result


@reference_implementation()
def bdtrc(k: float, n: float, p: float) -> float:
    """Binomial distribution survival function."""
    k, n = mp.floor(k), mp.floor(n)
    return mp.betainc(k + 1, n - k, 0, p, regularized=True)


@reference_implementation()
def bdtri(k: float, n: float, y: float) -> float:
    """Inverse function to `bdtr` with respect to `p`."""
    k, n = mp.floor(k), mp.floor(n)
    def f(p):
        return bdtr._mp(k, n, p) - y

    return solve_bisect(f, 0, 1)


@reference_implementation()
def bei(x: float) -> float:
    """Kelvin function bei."""
    return mp.bei(0, x)


@reference_implementation()
def beip(x: float) -> float:
    """Derivative of the Kelvin function bei."""
    return mp.diff(bei._mp, x, n=1)


@reference_implementation()
def ber(x: float) -> float:
    """Kelvin function ber."""
    return mp.ber(0, x)


@reference_implementation()
def berp(x: float) -> float:
    """Derivative of the Kelvin function ber."""
    return mp.diff(bei._mp, x, n=1)


@reference_implementation()
def besselpoly(a: float, lmb: float, nu: float) -> float:
    """Weighted integral of the Bessel function of the first kind."""
    def integrand(x):
        return x**lmb * mp.besselj(nu, 2 * a * x)

    return mp.quad(integrand, [0, 1])


@reference_implementation()
def beta(a: float, b: float) -> float:
    """Beta function."""
    return mp.beta(a, b)


@reference_implementation()
def betaln(a: float, b: float) -> float:
    """Natural logarithm of the absolute value of the Beta function."""
    return mp.log(abs(mp.beta(a, b)))


@reference_implementation()
def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete Beta function."""
    return mp.betainc(a, b, 0, x, regularized=True)


@reference_implementation()
def betaincc(a: float, b: float, x: float) -> float:
    """Complement of the regularized incomplete Beta function."""
    return mp.betainc(a, b, x, 1.0, regularized=True)


@reference_implementation()
def betaincinv(a: float, b: float, y: float) -> float:
    """Inverse of the regularized incomplete beta function."""
    def f(x):
        return mp.betainc(a, b, 0, x, regularized=True) - y

    return solve_bisect(f, 0, 1)


@reference_implementation()
def betainccinv(a: float, b: float, y: float) -> float:
    """Inverse of the complemented regularized incomplete beta function."""
    def f(x):
        return mp.betainc(a, b, x, 1.0, regularized=True) - y

    return solve_bisect(f, 0, 1)


@reference_implementation()
def binom(n: float, k: float) -> float:
    """Binomial coefficient considered as a function of two real variables."""
    return mp.binomial(n, k)


@reference_implementation()
def cbrt(x: float) -> float:
    """Cube root of x."""
    return mp.cbrt(x)


@reference_implementation(uses_mp=False)
def cem(m: float, q, float, x: float) -> Tuple[float, float]:
    """Even Mathieu function and its derivative."""
    return special.mathieu_cem(m, q, x)


@reference_implementation(uses_mp=False)
def cem_cva(m: float, q: float) -> float:
    """Characteristic value of even Mathieu functions."""
    return special.mathieu_a(m, q)


@reference_implementation()
def chdtr(v: float, x: float) -> float:
    """Chi square cumulative distribution function."""
    return mp.gammainc(v / 2, 0, x / 2, regularized=True)

@reference_implementation()
def chdtrc(v: float, x: float) -> float:
    """Chi square survival function."""
    return mp.gammainc(v / 2, x / 2, mp.inf, regularized=True)


@reference_implementation()
def chdtri(v: float, p: float) -> float:
    """Inverse to `chdtrc` with respect to `x`."""
    # TODO. Figure out why chdtri inverts chdtrc and not chdtr
    return 2 * gammainccinv._mp(v / 2, p)


@reference_implementation(uses_mp=False)
def cosdg(x: float) -> float:
    """Cosine of the angle x given in degrees."""
    return special.cosdg(x)


@reference_implementation()
def cosm1(x: float) -> float:
    """cos(x) - 1 for use when x is near zero."""
    # set the precision high enough to avoid catastrophic cancellation
    # cos(x) - 1 = x^2/2 + O(x^4) for x near 0
    precision = min(int(mp.ceil(-2*mp.log(abs(x), b=2))), 1024) + 53
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        result =  mp.cos(x) - mp.one
    return result


@overload
def cospi(x: float) -> float: ...
@overload
def cospi(x: complex) -> complex: ...


@reference_implementation()
def cospi(x):
    """Cosine of pi*x."""
    return mp.cospi(x)


@reference_implementation(uses_mp=False)
def cotdg(x: float) -> float:
    """Cotangent of the angle x given in degrees."""
    return special.cotdg(x)


@overload
def cyl_bessel_i(v: float, z: float) -> float: ...
@overload
def cyl_bessel_i(v: float, z:complex) -> complex: ...


def cyl_bessel_i(v, z):
    """Modified Bessel function of the first kind.


    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.besseli(v, z)


@reference_implementation()
def cyl_bessel_i0(z: float) -> float:
    """Modified Bessel function of order 0."""
    return mp.besseli(0, z)


@reference_implementation()
def cyl_bessel_i0e(z: float) -> float:
    """Exponentially scaled modified Bessel function of order 0."""
    return mp.exp(-abs(z.real)) * mp.besseli(0, z)


@reference_implementation()
def cyl_bessel_i1(z: float)-> float:
    """Modified Bessel function of order 1."""
    return mp.besseli(1, z)


@reference_implementation()
def cyl_bessel_i1e(z: float) -> float:
    """Exponentially scaled modified Bessel function of order 1."""
    return mp.exp(-abs(z.real)) * mp.besseli(1, z)


@overload
def cyl_bessel_ie(v: float, z: float) -> float: ...
@overload
def cyl_bessel_ie(v: float, z:complex) -> complex: ...


@reference_implementation()
def cyl_bessel_ie(v, x):
    """Exponentially scaled modified Bessel function of the first kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        w = z + mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.exp(-abs(z.real)) * mp.besseli(v, w)


@overload
def cyl_bessel_j(v: float, z: float) -> float: ...
@overload
def cyl_bessel_j(v: float, z: complex) -> complex: ...


@reference_implementation()
def cyl_bessel_j(v, z):
    """Bessel function of the first kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.besselj(v, z)


@reference_implementation()
def cyl_bessel_j0(x: float) -> float:
    """Bessel function of the first kind of order 0."""
    return mp.j0(x)


@reference_implementation()
def cyl_bessel_j1(x: float) -> float:
    """Bessel function of the first kind of order 1."""
    return mp.j1(x)


@overload
def cyl_bessel_je(v: float, z: float) -> float: ...
@overload
def cyl_bessel_je(v: float, z: complex) -> complex: ...


@reference_implementation()
def cyl_bessel_je(v, z):
    """Exponentially scaled Bessel function of the first kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        w = z + mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.exp(-abs(z.imag)) * mp.besselj(v, w)


@overload
def cyl_bessel_k(v: float, z: float) -> float: ...
@overload
def cyl_bessel_k(v: float, z: complex) -> complex: ...


@reference_implementation()
def cyl_bessel_k(v, z):
    """Modified Bessel function of the second kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.besselk(v, z)


@reference_implementation()
def cyl_bessel_k0(x: float) -> float:
    """Modified Bessel function of the second kinf of order 0."""
    return mp.besselk(0, x)


@reference_implementation()
def cyl_bessel_k0e(x: float) -> float:
    """Exponentially scaled modified Bessel function K of order 0."""
    return mp.exp(x) * mp.besselk(0, x)


@reference_implementation()
def cyl_bessel_k1(x: float) -> float:
    """Modified Bessel function of the second kind of order 0."""
    return mp.besselk(1, x)


@reference_implementation()
def cyl_bessel_k1e(x: float) -> float:
    """Exponentially scaled modified Bessel function K of order 1."""
    return mp.exp(x) * mp.besselk(1, x)


@overload
def cyl_bessel_ke(v: float, z: float) -> float: ...
@overload
def cyl_bessel_ke(v: float, z: complex) -> complex: ...


@reference_implementation()
def cyl_bessel_ke(v, z):
    """Exponentially scaled modified Bessel function of the second kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        w = z + mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.exp(z) * mp.besselk(v, w)


@overload
def cyl_bessel_y(v: float, z: float) -> float: ...
@overload
def cyl_bessel_y(v: float, z: complex) -> complex: ...


@reference_implementation()
def cyl_bessel_y(v, z):
    """Bessel function of the second kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.bessely(v, z)


@reference_implementation()
def cyl_bessel_y0(z: float) -> float:
    """Bessel function of the second kind of order 0."""
    return mp.bessely(0, z)


@reference_implementation()
def cyl_bessel_y1(z: float) -> float:
    """Bessel function of the second kind of order 1."""
    return mp.bessely(1, z)


@overload
def cyl_bessel_ye(v: float, z: float) -> float: ...
@overload
def cyl_bessel_ye(v: float, z: complex) -> complex: ...


@reference_implementation()
def cyl_bessel_ye(v, z):
    """Exponentially scaled Bessel function of the second kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        w = z + mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.bessely(w, z) * mp.exp(-abs(z.imag))


@overload
def cyl_hankel_1(v: float, z: float) -> float: ...
@overload
def cyl_hankel_1(v: float, z: complex) -> complex: ...


@reference_implementation()
def cyl_hankel_1(v, z):
    """Hankel function of the first kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.hankel1(v, z)


@overload
def cyl_hankel_1e(v: float, z: float) -> float: ...
@overload
def cyl_hankel_1e(v: float, z: complex) -> complex: ...


@reference_implementation()
def cyl_hankel_1e(v, z):
    """Exponentially scaled Hankel function of the first kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        w = z + mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.hankel1(v, w) * mp.exp(z * -1j)


@overload
def cyl_hankel_2(v: float, z: float) -> float: ...
@overload
def cyl_hankel_2(v: float, z: complex) -> complex: ...


@reference_implementation()
def cyl_hankel_2(v, z):
    """Hankel function of the second kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.hankel2(v, z)


@overload
def cyl_hankel_2e(v: float, z: float) -> float: ...
@overload
def cyl_hankel_2e(v: float, z: complex) -> complex: ...


@reference_implementation()
def cyl_hankel_2e(v, z):
    """Exponentially scaled Hankel function of the second kind.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        w = z + mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.hankel2(v, w) * mp.exp(z * 1j)


@overload
def dawsn(x: float) -> float: ...
@overload
def dawsn(x: complex) -> complex: ...


@reference_implementation()
def dawsn(x):
    """Dawson's integral

    dawsn is an entire function
    """
    def integrand(t):
        return mp.exp(t**2)

    return mp.exp(-x**2) * mp.quad(integrand, [0, x])


@overload
def digamma(x: float) -> float: ...
@overload
def digamma(x: complex) -> complex: ...


@reference_implementation()
def digamma(x):
    """The digamma function.

    Notes
    -----
    Poles at nonpositive integers.
    """
    return mp.digamma(x)


@reference_implementation()
def ellipe(m: float) -> float:
    """Complete elliptic integral of the second kind."""
    return mp.ellipe(m)


@reference_implementation()
def ellipeinc(phi: float, m: float) -> float:
    """Incomplete elliptic integral of the second kind."""
    return mp.ellipe(phi, m)


@reference_implementation()
def ellipj(u: float, m: float) -> Tuple[float, float, float, float]:
    """Jacobian Elliptic functions."""
    sn = mp.ellipfun("sn", u=u, m=m)
    cn = mp.ellipfun("cn", u=u, m=m)
    dn = mp.ellipfun("dn", u=u, m=m)
    phi = mp.asin(sn)
    return sn, cn, dn, phi


@reference_implementation()
def ellipk(m: float) -> float:
    """Complete elliptic integral of the first kind."""
    return mp.ellipk(m)


@reference_implementation()
def ellipkinc(phi: float, m: float) -> float:
    """Incomplete elliptic integral of the first kind."""
    return mp.ellipf(phi, m)


@reference_implementation()
def ellipkm1(p: float) -> float:
    """Complete elliptic integral of the first kind around m = 1."""
    with mp.workprec(max(mp.prec, int(mp.ceil(-mp.log(abs(p), b=2))))):
        # set the precision high enough that mp.one - p != 1
        result = mp.ellipk(1 - p)
    return result


@overload
def erf(x: float) -> float: ...
@overload
def erf(x: complex) -> complex: ...


@reference_implementation()
def erf(x):
    """Error function.

    erf is an entire function
    """
    return mp.erf(x)


@overload
def erfi(x: float) -> float: ...
@overload
def erfi(x: complex) -> complex: ...


@reference_implementation()
def erfi(x):
    """Imaginary error function.

    erfi is an entire function
    """
    return -mp.j * mp.erf(mp.j * x)


@overload
def erfc(x: float) -> float: ...
@overload
def erfc(x: complex) -> complex: ...


@reference_implementation()
def erfc(x):
    """Complementary error function 1 - erf(x).

    Notes
    -----
    erfc is an entire function
    """
    return mp.erfc(x)


@overload
def erfcx(x: float) -> float: ...
@overload
def erfcx(x: complex) -> complex: ...


@reference_implementation()
def erfcx(x):
    """Scaled complementary error function exp(x**2) * erfc(x)

    Notes
    -----
    erfcx is an entire function
    """
    return mp.exp(x**2) * mp.erfc(x)


@reference_implementation()
def erfcinv(x: float) -> float:
    """Inverse of the complementary error function."""
    if not 0 <= x <= 2:
        return math.nan
    if x == 0:
        return math.inf
    with mp.workprec(max(mp.prec, int(mp.ceil(-mp.log(abs(x), b=2))) + 53)):
        result = mp.erfinv(mp.one - x)
    return result


@overload
def exp1(x: float) -> float: ...
@overload
def exp1(x: complex) -> complex: ...


@reference_implementation()
def exp1(x):
    """Exponential integral E1.

    Notes
    -----
    Logarithmic singularity at x = 0 with branch cut on (-inf, 0).
    """
    if x.imag == 0 and x.real < 0:
        # On branch cut, choose branch based on sign of zero
        x += mp.mpc("0", "1e-1000000000") * math.copysign(x.imag)
    return mp.e1(x)


@overload
def exp10(x: float) -> float: ...
@overload
def exp10(x: complex) -> complex: ...


@reference_implementation()
def exp10(x):
    """Compute 10**x."""
    return mp.mpf(10) ** x


@overload
def exp2(x: float) -> float: ...
@overload
def exp2(x: complex) -> complex: ...


@reference_implementation()
def exp2(x):
    """Compute 2**x."""
    return mp.mpf(2) ** x


@overload
def expi(x: float) -> float: ...
@overload
def expi(x: complex) -> complex: ...


@reference_implementation()
def expi(x):
    """Exponential integral Ei.

    Notes
    -----
    Logarithmic singularity at x = 0 with branch cut on (-inf, 0).
    """
    if x.imag == 0 and x.real < 0:
        # On branch cut, choose branch based on sign of zero
        x += mp.mpc("0", "1e-1000000000") * math.copysign(x.imag)
    return mp.ei(x)


@reference_implementation()
def expit(x: float) -> float:
    """Expit (a.k.a logistic sigmoid)."""
    return mp.sigmoid(x)


@overload
def expm1(x: float) -> float: ...
@overload
def expm1(x: complex) -> complex: ...


@reference_implementation()
def expm1(x):
    """exp(x) - 1.

    Notes
    -----
    expm1 is an entire function
    """
    return mp.expm1(x)


@reference_implementation()
def expn(n: int, x: float) -> float:
    """Generalized exponential integral En."""
    return mp.expint(n, x)


@reference_implementation()
def exprel(x: float) -> float:
    """Relative error exponential, (exp(x) - 1)/x."""
    with mp.workprec(max(mp.prec, int(mp.ceil(-mp.log(abs(x), b=2))) + 53)):
        # set the precision high enough to avoid catastrophic cancellation
        # Near 0, mp.exp(x) - 1 = x + O(x^2)
        result = (mp.exp(x) - 1) / x
    return result


@reference_implementation()
def fdtr(dfn: float, dfd: float, x: float) -> float:
    """F cumulative distribution function."""
    x_dfn = x * dfn
    return mp.betainc(dfn / 2, dfd / 2, 0, x_dfn / (dfd + x_dfn), regularized=True)


@reference_implementation()
def fdtrc(dfn: float, dfd: float, x: float) -> float:
    """F survival function."""
    x_dfn = x * dfn
    return mp.betainc(dfn / 2, dfd / 2, x_dfn / (dfd + x_dfn), 1.0, regularized=True)


@reference_implementation()
def fdtri(dfn: float, dfd: float, p: float) -> float:
    """F cumulative distribution function."""
    q = betaincinv._mp(dfn / 2, dfd / 2, p)
    return q * dfd / ((1 - q) * dfn)

@overload
def fresnel(x: float) -> Tuple[float, float]: ...
@overload
def fresnel(x: complex) -> Tuple[complex, complex]: ...


@reference_implementation()
def fresnel(x):
    """Fresnel integrals.

    Notes
    -----
    Fresnel integrals are entire functions
    """
    return mp.fresnels(x), mp.fresnelc(x)


@overload
def gamma(x: float) -> float: ...
@overload
def gamma(x: complex) -> complex: ...


@reference_implementation()
def gamma(x):
    """Gamma function.

    Notes
    -----
    Poles at nonpositive integers
    """
    if x == 0.0:
        if isinstance(x, float):
            return math.copysign(mp.inf, x)
        return mp.nan
    if x.real < 0 and x.imag == 0 and x.real == int(x.real):
        return mp.nan
    return mp.gamma(x)


@reference_implementation()
def gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function."""
    return mp.gammainc(a, x, mp.inf, regularized=True)


@reference_implementation()
def gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function."""
    return mp.gammainc(a, 0, x, regularized=True)


def _gammainccinv_initial_bracket(a, y):
    g = mp.gamma(a)
    u = y * g
    if a == 1:
        initial_guess = -mp.log(u)
        return (
            math.nextafter(float(initial_guess), -math.inf),
            math.nextafter(float(initial_guess), math.inf),
        )
    # For a > 1 use the inequalities in DLMF 8.10.3 and invert the left and
    # right hand sides to get an initial bracket.
    # https://dlmf.nist.gov/8.10#E3
    z = -(u)**(1/(a-1))/(a - 1)
    if a > 1:
        h = ((a - 1) / mp.e)**(a - 1) / g
        if y > h:
            # Not invertible if y > h, but can use y == h to get an
            # upper bound.
            return (0, _gammainccinv_initial_bracket(a, h)[1])
        initial_guess = -(a - 1) * mp.lambertw(z, k=-1).real

        # Other side of inequality doesn't have a closed form. Use
        # secant method to invert it.
        def f(x): return x**(a-1)*mp.exp(-x)/g + (a - 1)/x - y

        try:
            guess2 = solve_secant(f, initial_guess, 1.01*initial_guess)
        except RuntimeError:
            # If the secant method failed, be conservative and pick right
            # endpoint larger than the largest double.
            guess2 = mp.mpf("2e308")

        return (initial_guess, guess2)
    # If a < 1, we can only get a reliable upper bound.
    initial_guess = -(a - 1) * mp.lambertw(z, k=0).real
    return (0, initial_guess)


@reference_implementation()
def gammainccinv(a: float, y: float) -> float:
    """Inverse to the regularized upper incomplete gamma function."""
    # special cases
    if y == 0:
        return math.inf
    if y == 1:
        return 0.0
    if y > 1 or y < 0:
        return math.nan

    def f(x):
        return mp.gammainc(a, x, mp.inf, regularized=True) - y

    xl, xr = _gammainccinv_initial_bracket(a, y)
    if xl >= xr or mp.sign(f(xl)) == mp.sign(f(xr)):
        # This should not happen, but is here so code reviewers won't need to
        # verify that _gammainccinv_initial_bracket will always return a
        # valid bracket.
        xl, xr = mp.zero, mp.mpf("2e308")
    return solve_bisect(f, xl, xr)


@reference_implementation()
def gammaincinv(a: float, y: float) -> float:
    """Inverse to the regularized lower incomplete gamma function."""
    with mp.workprec(max(mp.prec, int(mp.ceil(-mp.log(abs(y), b=2))) + 53)):
        # set the precision high enough to resolve mp.one - y
        result = gammainccinv._mp(a, mp.one - y)
    return result


@reference_implementation()
def gammaln(x: float) -> float:
    """Logarithm of the absolute value of the gamma function."""
    if x.real <= 0 and x == int(x):
        return mp.inf
    return mp.log(abs(mp.gamma(x)))


@reference_implementation()
def gammasgn(x: float) -> float:
    """Sign of the gamma function."""
    if x == 0.0:
        return math.copysign(1.0, x)
    if x < 0 and x == int(x):
        return mp.nan
    return mp.sign(mp.gamma(x))


@reference_implementation()
def gdtr(a: float, b: float, x: float) -> float:
    """Gamma distribution cumulative distribution function."""
    return mp.gammainc(b, 0, a * x, regularized=True)


@reference_implementation()
def gdtrc(a: float, b: float, x: float)-> float:
    """Gamma distribution survival function."""
    return mp.gammainc(b, a * x, mp.inf, regularized=True)


@reference_implementation(uses_mp=False)
def gdtrib(a: float, p: float, x: float) -> float:
    """Inverse of `gdtr` vs b."""
    return special.gdtrib(a, p, x)


@overload
def hyp1f1(a: float, b: float, z: float) -> float: ...
@overload
def hyp1f1(a: float, b: float, z: complex) -> complex: ...


@reference_implementation()
def hyp1f1(a, b, z):
    """Confluent hypergeometric function 1F1.

    Notes
    -----
    Entire in a and z
    Meromorphic in b with poles at nonpositive integers
    """
    return mp.hyp1f1(a, b, z)


@overload
def hyp2f1(a: float, b: float, c: float, z: float) -> float: ...
@overload
def hyp2f1(a: float, b: float, c: float, z: complex) -> complex: ...


@reference_implementation()
def hyp2f1(a, b, c, z):
    """Gauss hypergeometric function 2F1(a, b; c; z).

    Notes
    -----
    Branch point at ``z=1`` with branch cut along ``(1, inf)``.
    """
    if z.imag == 0 and z.real > 1:
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.hyp2f1(a, b, c, z)


@reference_implementation()
def hyperu(a: float, b: float, z: float) -> float:
    """Confluent hypergeometric function U.

    Notes
    -----
    Branch point at ``z=0`` with branch cut along ``(-inf, 0)``.
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000000") * math.copysign(z.imag)
    return mp.hyperu(a, b, z)


@reference_implementation()
def it1i0k0(x: float) -> Tuple[float, float]:
    """Integrals of modified Bessel functions of order 0."""
    result1 = mp.quad(cyl_bessel_i0._mp, [0, x])
    result2 = mp.quad(cyl_bessel_k0._mp, [0, x])
    return result1, result2


@reference_implementation()
def it1j0y0(x: float) -> Tuple[float, float]:
    """Integrals of Bessel functions of the first kind of order 0."""
    result1 = mp.quad(cyl_bessel_j0._mp, [0, x])
    result2 = mp.quad(cyl_bessel_y0._mp, [0, x])
    return result1, result2


@reference_implementation()
def it2i0k0(x: float) -> Tuple[float, float]:
    """Integrals related to modified Bessel functions of order 0.

    TODO: Take a closer look at this and it2j0y0.
    """

    def f1(t):
        return (cyl_bessel_i0._mp(t) - 1) / t

    def f2(t):
        return cyl_bessel_k0._mp(t) / t

    result1 = mp.quad(f1, [0, x])
    result2 = mp.quad(f2, [0, x])
    return result1, result2


@reference_implementation()
def it2j0y0(x: float) -> Tuple[float, float]:
    """Integrals related to Bessel functions of the first kind of order 0."""

    def f1(t):
        return (1 - cyl_bessel_j0._mp(t)) / t

    def f2(t):
        return cyl_bessel_y0._mp(t) / t

    result1 = mp.quad(f1, [0, x])
    result2 = mp.quad(f2, [0, x])
    return result1, result2


@reference_implementation()
def it2struve0(x: float) -> float:
    """Integral related to the Struve function of order 0."""
    def f(t):
        return struve_h._mp(0, t) / t

    return mp.quad(f, [0, x])


@reference_implementation()
def itairy(x: float) -> Tuple[float, float, float, float]:
    """Integrals of Airy functions."""
    def ai(t):
        return mp.airyai(t)

    def bi(t):
        return mp.airybi(t)

    result1 = mp.quad(ai, [0, x])
    result2 = mp.quad(bi, [0, x])
    result3 = mp.quad(ai, [-x, 0])
    result4 = mp.quad(bi, [-x, 0])
    return result1, result2, result3, result4


@reference_implementation()
def itmodstruve0(x: float) -> float:
    """Integral of the modified Struve function of order 0."""
    def f(t):
        return struve_l._mp(0, t)

    return mp.quad(f, [0, x])


@reference_implementation()
def itstruve0(x: float) -> float:
    """Integral of the modified Struve function of order 0."""
    def f(t):
        return struve_l._mp(0, t)

    return mp.quad(f, [0, x])


@reference_implementation()
def iv_ratio(v: float, x: float) -> float:
    """Returns the ratio ``iv(v, x) / iv(v - 1, x)``"""
    numerator = cyl_bessel_i._mp(v, x)
    return numerator / cyl_bessel_i._mp(v - 1, x)


@reference_implementation()
def iv_ratio_c(v: float, x: float) -> float:
    """Returns ``1 - iv_ratio(v, x)``."""
    numerator = cyl_bessel_i._mp(v, x)
    denominator = cyl_bessel_i._mp(v - 1, x)
    # Set precision high enough to avoid catastrophic cancellation.
    # For large x, iv_ratio_c(v, x) ~ (v - 0.5) / x
    if x != 0:
        precision = int(mp.ceil(-mp.log(abs((v - 0.5)/x), b=2))) + 53
        precision = max(mp.prec, precision)
    else:
        precision = mp.prec
    with mp.workprec(precision):
        result = mp.one - numerator / denominator
    return result


@reference_implementation()
def kei(x: float) -> float:
    """Kelvin function kei."""
    return mp.kei(0, x)


@reference_implementation()
def keip(x: float) -> float:
    """Derivative of the Kelvin function kei."""
    return mp.diff(kei._mp, x, n=1)


@reference_implementation(uses_mp=False)
def kelvin(x: float) -> Tuple[complex, complex, complex, complex]:
    """Kelvin functions as complex numbers."""
    be = complex(ber(x), bei(x))
    ke = complex(ker(x), kei(x))
    bep = complex(berp(x), beip(x))
    kep = complex(kerp(x), keip(x))
    return be, ke, bep, kep


@reference_implementation()
def ker(x: float) -> float:
    """Kelvin function ker."""
    return mp.ker(0, x)


@reference_implementation()
def kerp(x: float) -> float:
    """Derivative of the Kelvin function kerp."""
    return mp.diff(ker._mp, x, n=1)


@reference_implementation(uses_mp=False)
def kolmogc(x: float) -> float:
    """CDF of Kolmogorov distribution.

    CDF of Kolmogorov distribution can be expressed in terms of
    Jacobi Theta functions.
    TODO: Look into writing arbitrary precision reference implementations
    for kolmogc, kolmogci, kolmogi, and kolmogorov.
    """
    return special._ufuncs._kolmogc(x)


@reference_implementation(uses_mp=False)
def kolmogci(x: float) -> float:
    """Inverse CDF of Kolmogorov distribution."""
    return special._ufuncs._kolmogci(x)


@reference_implementation(uses_mp=False)
def kolmogi(x: float) -> float:
    """Inverse Survival Function of Kolmogorov distribution."""
    return special._ufuncs._kolmogi(x)


@reference_implementation(uses_mp=False)
def kolmogorov(x: float) -> float:
    """Survival Function of Kolmogorov distribution."""
    return special._ufuncs._kolmogorov(x)


@reference_implementation(uses_mp=False)
def kolmogp(x: float) -> float:
    """Negative of PDF of Kolmogorov distribution.

    TODO: Why is this the negative pdf?
    """
    return special._ufuncs._kolmogp(x)


@reference_implementation()
def lambertw(z: complex, k: int) -> complex:
    """Lambert W function.

    Notes
    -----
    Branch cut on (-inf, 0). k = 0 corresponds to the principle
    branch. There are infinitely many branches.
    """
    if z.imag == 0 and (z.real < 0 and k !=0 or z.real < -1/mp.e):
        # On branch cut, choose branch based on sign of zero.
        # mpmath's lambertw currently converts z to a complex128 internally,
        # so the small step here can't be smaller than the smallest subnormal.
        z += mp.mpc(0, 5e-324) * math.copysign(z.imag)
    return mp.lambertw(z, k=k)


@reference_implementation()
def lanczos_sum_expg_scaled(z: float) -> float:
    """Exponentially scaled Lanczos approximation to the Gamma function."""
    g = mp.mpf("6.024680040776729583740234375")
    return (mp.e / (z + g - 0.5)) ** (z - 0.5) * mp.gamma(z)


@reference_implementation()
def lgam1p(x: float) -> float:
    """Logarithm of gamma(x + 1)."""
    return mp.log(mp.gamma(x + 1))


@overload
def log1p(z: float) -> float: ...
@overload
def log1p(z: complex) -> complex: ...


@reference_implementation()
def log1p(x):
    """Logarithm of x + 1.

    Notes
    -----
    Branch cut on (-inf, -1)
    """
    if z.imag == 0 and z.real < -1:
        # On branch cut, choose branch based on sign of zero.
        z += mp.mpc(0, "1e-1000000000") * math.copysign(z.imag)
    return mp.log1p(x)


@overload
def log1pmx(z: float) -> float: ...
@overload
def log1pmx(z: complex) -> complex: ...


@reference_implementation()
def log1pmx(z):
    """log(x + 1) - x.

    Notes
    -----
    Branch cut on (-inf, -1)
    """
    if z.imag == 0 and z.real < -1:
        # On branch cut, choose branch based on sign of zero.
        z += mp.mpc(0, "1e-1000000000") * math.copysign(z.imag)
    # set the precision high enough to avoid catastrophic cancellation.
    # Near z = 0 log(1 + z) - z = -z^2/2 + O(z^3)
    precision = min(int(mp.ceil(-2*mp.log(abs(x), b=2))), 1024) + 53
    precision = max(mp.prec, precision)
    with mp.workprec(precision):
        result = mp.log1p(z) - z
    return result


@overload
def loggamma(z: float) -> float: ...
@overload
def loggamma(z: complex) -> complex: ...


@reference_implementation()
def loggamma(z):
    """Principal branch of the logarithm of the gamma function.

    Notes
    -----
    Logarithmic singularity at z = 0
    Branch cut on (-inf, 0).
    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero.
        z += mp.mpc(0, "1e-1000000000") * math.copysign(z.imag)
    return mp.loggamma(z)


@reference_implementation()
def log_expit(x: float) -> float:
    """Log of `expit`."""
    return mp.log(mp.sigmoid(x))


@reference_implementation()
def log_wright_bessel(a: float, b: float, x: float) -> float:
    """Natural logarithm of Wright's generalized Bessel function."""
    return mp.log(_wright_bessel(a, b, x))


@reference_implementation()
def logit(x: float) -> float:
    """Logit function ``logit(p) = log(p/(1 - p))``"""
    with mp.workprec(max(mp.prec, int(mp.ceil(-mp.log(abs(p), b=2))) + 53)):
        # set the precision high enough to resolve 1 - p.
        result = mp.log(p/(1-p))
    return result


@reference_implementation(uses_mp=False)
def mcm1(m: float, q: float, x: float) -> Tuple[float, float]:
    """Even modified Mathieu function of the first kind and its derivative."""
    return special.mathieu_modcem1(m, q, x)


@reference_implementation(uses_mp=False)
def mcm2(m: float, q: float, x: float) -> Tuple[float, float]:
    """Even modified Mathieu function of the second kind and its derivative."""
    return special.mathieu_modcem2(m, q, x)


@reference_implementation(uses_mp=False)
def modified_fresnel_minus(x: float) -> Tuple[complex, complex]:
    """Modified Fresnel negative integrals."""
    return special.modfresnelm(x)


@reference_implementation(uses_mp=False)
def modified_fresnel_plus(x: float) -> Tuple[complex, complex]:
    """Modified Fresnel negative integrals."""
    return special.modfresnelp(x)


@reference_implementation(uses_mp=False)
def msm1(m: float, q: float, x: float) -> Tuple[float, float]:
    """Odd modified Mathieu function of the first kind and its derivative."""
    return special.mathieu_modsem1(m, q, x)


@reference_implementation(uses_mp=False)
def msm2(m: float, q: float, x: float) -> Tuple[float, float]:
    """Odd modified Mathieu function of the second kind and its derivative."""
    return special.mathieu_modsem2(m, q, x)


@reference_implementation()
def nbdtr(k: int, n: int, p: float) -> float:
    """Negative binomial cumulative distribution function."""
    return mp.betainc(n, k + 1, 0, p)


@reference_implementation()
def nbdtrc(k: int, n: int, p: float) -> float:
    """Negative binomial survival function."""
    return mp.betainc(n, k + 1, p, 1)


@reference_implementation()
def ndtr(x: float) -> float:
    """Cumulative distribution of the standard normal distribution."""
    return mp.ncdf(x)


@reference_implementation()
def ndtri(y: float) -> float:
    """Inverse of `ndtr` vs x."""
    if not 0 <= y <= 1:
        return math.nan
    with mp.workprec(max(mp.prec, int(mp.ceil(-mp.log(abs(2*y), b=2))) + 53)):
        # set the precision high enough to resolve 2*y - 1
        result = mp.sqrt(2) * mp.erfinv(2*y - 1)
    return result


@reference_implementation(uses_mp=False)
def oblate_aswfa(
    m: float, n: float, c: float, cv: float, x: float
) -> Tuple[float, float]:
    """Oblate spheroidal angular function obl_ang1 for precomputed cv

    cv: Characteristic Value
    """
    return special.obl_ang1_cv(m, n, c, cv, x)


@reference_implementation(uses_mp=False)
def oblate_aswfa_nocv(
    m: float, n: float, c: float, x: float
) -> Tuple[float, float]:
    """Oblate spheroidal angular function of the first kind and its derivative."""
    return special.obl_ang1(m, n, c, x)


@reference_implementation(uses_mp=False)
def oblate_radial1(
    m: float, n: float, c: float, cv: float, x: float
) -> Tuple[float, float]:
    """Oblate spheroidal radial function obl_rad1 for precomputed cv

    cv: Characteristic Value
    """
    return special.obl_rad1_cv(m, n, c, cv, x)


@reference_implementation(uses_mp=False)
def oblate_radial1_nocv(
    m: float, n: float, c: float, x: float
) -> Tuple[float, float]:
    """Oblate spheroidal radial function of the first kind and its derivative."""
    return special.obl_rad1(m, n, c, x)


@reference_implementation(uses_mp=False)
def oblate_radial2(
    m: float, n: float, c: float, cv: float, x: float
) -> Tuple[float, float]:
    """Oblate spheroidal angular function obl_rad2 for precomputed cv

    cv: Characteristic Value
    """
    return special.obl_rad2_cv(m, n, c, cv, x)


@reference_implementation(uses_mp=False)
def oblate_radial2_nocv(
    m: float, n: float, c: float, x: float
) -> Tuple[float, float]:
    """Oblate spheroidal radial function of the second kind and its derivative."""
    return special.obl_rad2(m, n, c, x)


@reference_implementation(uses_mp=False)
def oblate_segv(m: float, n: float, c: float) -> float:
    """Characteristic value of oblate spheroidal function."""
    return special.obl_cv(m, n, c)


@reference_implementation()
def owens_t(h: float, a: float) -> float:
    """Owen's T Function."""
    def integrand(x):
        return mp.exp(-(h**2) * (1 + x**2) / 2) / (1 + x**2)

    return mp.quad(integrand, [0, a]) / (2 * mp.pi)


@reference_implementation()
def pbdv(v: float, x: float) -> Tuple[float, float]:
    """Parabolic cylinder function D."""
    d = mp.pcfd(v, x)
    dp = mp.diff(lambda t: mp.pcfd(v, t), x)
    return d, dp


@reference_implementation()
def pbvv(v: float, x: float) -> Tuple[float, float]:
    """Parabolic cylinder function V."""
    with mp.workprec(max(mp.prec, int(mp.ceil(-mp.log(abs(2*v), b=2))) + 53)):
        # Set precision to guarantee -v - 0.5 retains precision for very small v.
        d = mp.pcfv(-v - 0.5, x)
        dp = mp.diff(lambda t: mp.pcfv(-v - 0.5, t), x)
    return d, dp


@reference_implementation()
def pbwa(v: float, x: float) -> Tuple[float, float]:
    """Parabolic cylinder function W."""
    d = mp.pcfw(v, x)
    dp = mp.diff(lambda t: mp.pcfw(v, t), x)
    return d, dp


@reference_implementation()
def pdtr(k: float, m: float) -> float:
    """Poisson cumulative distribution function."""
    k = mp.floor(k)
    return gammaincc._mp(k + 1, m)


@reference_implementation()
def pdtrc(k: float, m: float) -> float:
    """Poisson survival function."""
    k = mp.floor(k)
    return gammainc._mp(k + 1, m)


@reference_implementation()
def pdtri(k: float, y: float) -> float:
    """Inverse of `pdtr` vs m."""
    k = mp.floor(k)
    return gammainccinv._mp(k + 1, y)


@reference_implementation(uses_mp=False)
def pmv(m: int, v: float, x: float) -> float:
    """Associated Legendre function of integer order and real degree."""
    return special.lpmv(m, v, x)


@reference_implementation()
def poch(z: float, m: float) -> float:
    """Pochhammer symbol."""
    return mp.rf(z, m)


@reference_implementation(uses_mp=False)
def prolate_aswfa(
    m: float, n: float, c: float, cv: float, x: float
) -> Tuple[float, float]:
    """Prolate spheroidal angular function pro_ang1 for precomputed cv

    cv: Characteristic Value
    """
    return special.pro_ang1_cv(m, n, c, cv, x)


@reference_implementation(uses_mp=False)
def prolate_aswfa_nocv(
    m: float, n: float, c: float, x: float
) -> Tuple[float, float]:
    """Prolate spheroidal angular function of the first kind and its derivative."""
    return special.pro_ang1(m, n, c, x)


@reference_implementation(uses_mp=False)
def prolate_radial1(
    m: float, n: float, c: float, cv: float, x: float
) -> Tuple[float, float]:
    """Prolate spheroidal radial function pro_rad1 for precomputed cv

    cv: Characteristic Value
    """
    return special.pro_rad1_cv(m, n, c, cv, x)


@reference_implementation(uses_mp=False)
def prolate_radial1_nocv(
    m: float, n: float, c: float, x: float
) -> Tuple[float, float]:
    """Prolate spheroidal radial function of the first kind and its derivative."""
    return special.pro_rad1(m, n, c, x)


@reference_implementation(uses_mp=False)
def prolate_radial2(
    m: float, n: float, c: float, cv: float, x: float
) -> Tuple[float, float]:
    """Prolate spheroidal angular function pro_rad2 for precomputed cv

    cv: Characteristic Value
    """
    return special.pro_rad2_cv(m, n, c, cv, x)


@reference_implementation(uses_mp=False)
def prolate_radial2_nocv(
    m: float, n: float, c: float, x: float
) -> Tuple[float, float]:
    """Prolate spheroidal radial function of the second kind and its derivative."""
    return special.pro_rad2(m, n, c, x)


@reference_implementation(uses_mp=False)
def prolate_segv(m: float, n: float, c: float) -> float:
    """Characteristic value of prolate spheroidal function."""
    return special.pro_cv(m, n, c)


@reference_implementation()
def radian(d: float, m: float, s: float) -> float:
    """Convert from degrees to radians."""
    return mp.radians(d + m / 60 + s / 3600)


@overload
def rgamma(z: float) -> float: ...
@overload
def rgamma(z: complex) -> complex: ...


@reference_implementation()
def rgamma(z):
    """Reciprocal of the gamma function."""
    if x == 0.0:
        return x
    if x < 0 and x == int(x):
        return 0.0
    return mp.one / mp.gamma(z)


@overload
def riemann_zeta(z: float) -> float: ...
@overload
def riemann_zeta(z: complex) -> complex: ...


@reference_implementation()
def riemann_zeta(z):
    """Riemann zeta function.

    Notes
    -----
    A single pole at z = 1
    """
    if z == 1.0:
        return mp.nan
    return mp.zeta(z)


@reference_implementation()
def round(x):
    """Round to the nearest integer."""
    return mp.nint(x)


@reference_implementation()
def scaled_exp1(x: float) -> float:
    """Exponentially scaled exponential integral E1."""
    return mp.exp(x) * x * mp.e1(x)


@reference_implementation(uses_mp=False)
def sem(m: float, q, float, x: float) -> Tuple[float, float]:
    """Odd Mathieu function and its derivative."""
    return special.mathieu_sem(m, q, x)


@reference_implementation(uses_mp=False)
def sem_cva(m: float, q: float) -> float:
    """Characteristic value of odd Mathieu functions."""
    return special.mathieu_b(m, q)


@overload
def shichi(x: float) -> Tuple[float, float]: ...
@overload
def shichi(x: complex) -> Tuple[complex, complex]: ...


@reference_implementation()
def shichi(x):
    """Hyperbolic sine and cosine integrals.

    Notes
    -----
    Hyperbolic sine and cosine integrals are entire functions

    """
    return mp.shi(x), mp.chi(x)


@overload
def sici(x: float) -> Tuple[float, float]: ...
@overload
def sici(x: complex) -> Tuple[complex, complex]: ...


@reference_implementation()
def sici(x):
    """Sine and cosine integrals.

    Notes
    -----
    Sine and cosine integrals are entire functions
    """
    return mp.si(x), mp.ci(x)


@reference_implementation(uses_mp=False)
def sindg(x):
    """Sine of the angle `x` given in degrees."""
    return special.sindg(x)


@overload
def sinpi(x: float) -> float: ...
@overload
def sinpi(x: complex) -> complex: ...


@reference_implementation()
def sinpi(x):
    """Sine of pi*x.

    Note
    ----
    sinpi is an entire function
    """
    return mp.sinpi(x)


@reference_implementation(uses_mp=False)
def smirnov(n: int, d: float) -> float:
    """Kolmogorov-Smirnov complementary cumulative distribution function."""
    return special.smirnov(n, d)


@reference_implementation(uses_mp=False)
def smirnovc(n: int, d: float) -> float:
    """Kolmogorov-Smirnov cumulative distribution function."""
    return special._ufuncs.smirnovc(n, d)


@reference_implementation(uses_mp=False)
def smirnovc(n: int, d: float) -> float:
    """Kolmogorov-Smirnov cumulative distribution function."""
    return special._ufuncs.smirnovc(n, d)


@reference_implementation(uses_mp=False)
def smirnovci(n: int, p: float) -> float:
    """Inverse to `smirnovc`."""
    return special._ufuncs.smirnovc(n, p)


@reference_implementation(uses_mp=False)
def smirnovi(n: int, p: float) -> float:
    """Inverse to `smirnov`."""
    return special.smirnovi(n, p)


@reference_implementation(uses_mp=False)
def smirnovp(n: int, d: float) -> float:
    """Negative of Kolmogorov-Smirnov pdf."""
    return special._ufuncs._smirnovp(n, d)


@reference_implementation()
def spence(z: float) -> float:
    """Spence's function, also known as the dilogarithm."""
    with mp.workprec(max(mp.prec, int(mp.ceil(-mp.log(abs(z), b=2))) + 53)):
        # set the precision high enough that mp.one - z != 1
        result = mp.polylog(2, mp.one - z)
    return result


@reference_implementation()
def struve_h(v: float, x: float) -> float:
    """Struve function."""
    return mp.struveh(v, x)


@reference_implementation()
def struve_l(v: float, x: float) -> float:
    """Modified Struve function."""
    return  mp.struvel(v, x)


@reference_implementation(uses_mp=False)
def tandg(x: float) -> float:
    """Tangent of angle x given in degrees."""
    return special.tandg(x)


@reference_implementation()
def voigt_profile(x: float, sigma: float, gamma: float) -> float:
    """Voigt profile"""
    z = (x + mp.j *gamma) / (mp.sqrt(2) * sigma)
    w = mp.exp(-z**2) * mp.erfc(-mp.j * z)
    return w.real / (sigma * mp.sqrt(2*mp.pi))


@overload
def wofz(x: float) -> float: ...
@overload
def wofz(x: complex) -> complex: ...


@reference_implementation()
def wofz(x):
    """Faddeeva function

    Notes
    -----
    wofz is an entire function
    """
    return mp.exp(-x**2) * mp.erfc(-mp.j * x)


@reference_implementation()
def wright_bessel(a: float, b: float, x: float) -> float:
    """Wright's generalized Bessel function."""
    return _wright_bessel(a, b, x)


@overload
def xlogy(x: float, y: float) -> float: ...
@overload
def xlogy(x: complex, y: float) -> complex: ...


@reference_implementation()
def xlogy(x, y):
    """Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.

    Notes
    -----
    Branch cut on (-inf, 0)

    """
    if z.imag == 0 and z.real < 0:
        # On branch cut, choose branch based on sign of zero.
        z += mp.mpc(0, "1e-1000000000") * math.copysign(z.imag)
    if x == 0 and not (math.isnan(x.real) or math.isnan(x.imag)):
        return 0
    return x * mp.log(y)


@overload
def xlog1py(x: float, y: float) -> float: ...
@overload
def xlog1py(x: complex, y: float) -> complex: ...


@reference_implementation()
def xlog1py(x, y):
    """Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.

    Notes
    -----
    Branch cut on (-inf, 0)

    """
    if z.imag == 0 and z.real < -1:
        # On branch cut, choose branch based on sign of zero.
        z += mp.mpc(0, "1e-1000000000") * math.copysign(z.imag)
    if x == 0 and not (math.isnan(x.real) or math.isnan(x.imag)):
        return 0
    return x * mp.log1p(y)


@reference_implementation()
def zeta(z: float, q: float) -> float:
    """Hurwitz zeta function."""
    if z == 1.0:
        return mp.nan
    return mp.zeta(z, a=q)


@reference_implementation()
def zetac(z: float) -> float:
    """Riemann zeta function minus 1."""
    if z == 1.0:
        return mp.nan
    # set the precision high enough to avoid catastrophic cancellation.
    # As z approaches +inf in the right halfplane:
    # zeta(z) - 1 = 2^-z + O(3^-z).

    with mp.workprec(max(mp.prec, int(mp.ceil(z.real)) + 53)):

        result = mp.zeta(z) - mp.one
    return result


def _wright_bessel(a, b, x):
    def term(k):
        return x**k / (mp.factorial(k) * mp.gamma(a * k + b))

    return mp.nsum(term, [0, mp.inf])


def solve_bisect(f, xl, xr):
    if not xl < xr:
        xl, xr = xr, xl
    fl, fr = f(xl), f(xr)
    if fl == 0:
        return xl
    if fr == 0:
        return xr
    if mp.sign(fl) == mp.sign(fr):
        raise ValueError("f(xl) and f(xr) must have different signs")

    DBL_MAX = sys.float_info.max
    DBL_TRUE_MIN = 5e-324

    # Special handling for case where initial interval contains 0. It
    # can take a long time to find a root near zero to a given
    # relative tolerance through bisection alone, so this makes an
    # effort to find a better starting bracket.
    if xl <= 0 <= xr:
        f0 = f(0)
        if f0 == 0:
            return mp.zero
        vals = np.asarray([1e-50, 1e-100, 1e-150, 1e-200, 1e-250, 1e-300, 5e-324])
        if mp.sign(f0) == mp.sign(fr):
            vals = -vals
            for t in vals:
                if xl > t:
                    continue
                ft = f(t)
                if  mp.sign(ft) != mp.sign(fl):
                    xr = t;
                    break
                xl = t
        else:
            for t in vals:
                if xr < t:
                    continue
                ft = f(t)
                if  mp.sign(ft) != mp.sign(fr):
                    xl = t;
                    break
                xr = t

    iterations = Bisection(mp, f, [xl, xr])
    x_prev = mp.inf
    for x, error in iterations:
        if abs(x - x_prev) < abs(x)*1e-17:
            break
        if x < DBL_TRUE_MIN:
            return mp.zero
        if x > DBL_MAX:
            return mp.inf
        x_prev = x
    return x


def solve_secant(f, x0, x1, *, maxiter=10000):
    iterations = Secant(mp, f, [x0, x1])
    x_prev = x0
    for i, (x, error) in enumerate(iterations):
        if i >= maxiter:
            raise ValueError("maxiter exceeded")
        if abs(x - x_prev) < abs(x)*1e-17:
            break
    return x


_exclude = [
    "Bisection",
    "Secant",
    "math",
    "mp",
    "np",
    "overload",
    "reference_implementation",
    "scipy",
    "solve_bisect",
    "solve_secant",
    "special",
    "sys",
    "Tuple",
    "version",
]

__all__ = [s for s in dir() if not s.startswith("_") and s not in _exclude]
