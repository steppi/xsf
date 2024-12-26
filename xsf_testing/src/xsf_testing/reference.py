import math

from mpmath import mp  # type: ignore
from typing import overload, Tuple

from xsf_testing.util import reference_implementation


@overload
def airy(x: float) -> Tuple[float, float, float, float]: ...
@overload
def airy(x: complex) -> Tuple[complex, complex, complex, complex]: ...


@reference_implementation
def airy(z):
    """Airy functions and their derivatives."""
    ai = mp.airyai(z)
    aip = mp.airyai(z, derivative=1)
    bi = mp.airybi(z)
    bip = mp.airybi(z, derivative=1)
    return ai, aip, bi, bip


@overload
def airye(x: float) -> Tuple[float, float, float, float]: ...
@overload
def airye(x: complex) -> Tuple[complex, complex, complex, complex]: ...


@reference_implementation
def airye(z):
    """Exponentially scaled Airy functions and their derivatives."""
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


@reference_implementation
def bei(x: float) -> float:
    """Kelvin function bei."""
    return mp.bei(0, x)


@reference_implementation
def ber(x: float) -> float:
    """Kelvin function ber."""
    return mp.ber(0, x)


@reference_implementation
def besselpoly(a: float, lmb: float, nu: float) -> float:
    """Weighted integral of the Bessel function of the first kind."""
    def integrand(x):
        return x**lmb * mp.besselj(nu, 2 * a * x)

    return mp.quad(integrand, [0, 1])


@reference_implementation
def beta(a: float, b: float) -> float:
    """Beta function."""
    return mp.beta(a, b)


@reference_implementation
def betaln(a: float, b: float) -> float:
    """Natural logarithm of the absolute value of the Beta function."""
    return mp.log(abs(mp.beta(a, b)))


@reference_implementation
def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete Beta function."""
    return mp.betainc(a, b, 0, x, regularized=True)


@reference_implementation
def betaincc(a: float, b: float, x: float) -> float:
    """Complement of the regularized incomplete Beta function."""
    return mp.betainc(a, b, x, 1.0, regularized=True)


@reference_implementation
def bdtr(k: float, n: float, p: float) -> float:
    """Binomial distribution cumulative distribution function."""
    k, n = mp.floor(k), mp.floor(n)
    return mp.betainc(n - k, k + 1, 0, 1 - p, regularized=True)


@reference_implementation
def bdtrc(k: float, n: float, p: float) -> float:
    """Binomial distribution survival function."""
    k, n = mp.floor(k), mp.floor(n)
    return mp.betainc(k + 1, n - k, 0, p, regularized=True)


@reference_implementation
def binom(n: float, k: float) -> float:
    """Binomial coefficient considered as a function of two real variables."""
    return mp.binomial(n, k)


@reference_implementation
def cbrt(x: float) -> float:
    """Cube root of x."""
    return mp.cbrt(x)


@reference_implementation
def chdtr(v: float, x: float) -> float:
    """Chi square cumulative distribution function."""
    return mp.gammainc(v / 2, 0, x / 2, regularized=True)

@reference_implementation
def chdtrc(v: float, x: float) -> float:
    """Chi square survival function."""
    return mp.gammainc(v / 2, x / 2, mp.inf, regularized=True)


@reference_implementation
def cosdg(x: float) -> float:
    """Cosine of the angle x given in degrees."""
    return mp.cos(mp.radians(x))


@reference_implementation
def cosm1(x: float) -> float:
    """cos(x) - 1 for use when x is near zero."""
    return mp.cos(x) - 1


@overload
def cospi(x: float) -> float: ...
@overload
def cospi(x: complex) -> complex: ...


@reference_implementation
def cospi(x):
    """Cosine of pi*x."""
    return mp.cospi(x)


@reference_implementation
def cotdg(x: float) -> float:
    """Cotangent of the angle x given in degrees."""
    return mp.cot(mp.radians(x))


@reference_implementation
def cyl_bessel_i0(z: float) -> float:
    """Modified Bessel function of order 0."""
    return mp.besseli(0, z)


@reference_implementation
def cyl_bessel_i0e(z: float) -> float:
    """Exponentially scaled modified Bessel function of order 0."""
    return mp.exp(-abs(z.real)) * mp.besseli(0, z)


@reference_implementation
def cyl_bessel_i1(z: float)-> float:
    """Modified Bessel function of order 1."""
    return mp.besseli(1, z)


@reference_implementation
def cyl_bessel_i1e(z: float) -> float:
    """Exponentially scaled modified Bessel function of order 1."""
    return mp.exp(-abs(z.real)) * mp.besseli(1, z)


@overload
def cyl_bessel_i(v: float, z: float) -> float: ...
@overload
def cyl_bessel_i(v: float, z:complex) -> complex: ...


def cyl_bessel_i(v, z):
    """Modified Bessel function of the first kind."""
    return mp.besseli(v, z)


@overload
def cyl_bessel_ie(v: float, z: float) -> float: ...
@overload
def cyl_bessel_ie(v: float, z:complex) -> complex: ...


@reference_implementation
def cyl_bessel_ie(v, x):
    """Exponentially scaled modified Bessel function of the first kind."""
    return mp.exp(-abs(z.real)) * mp.besseli(v, z)


@reference_implementation
def cyl_bessel_j0(x: float) -> float:
    """Bessel function of the first kind of order 0."""
    return mp.j0(x)


@reference_implementation
def cyl_bessel_j1(x: float) -> float:
    """Bessel function of the first kind of order 1."""
    return mp.j1(x)


@overload
def cyl_bessel_j(v: float, z: float) -> float: ...
@overload
def cyl_bessel_j(v: float, z: complex) -> complex: ...


@reference_implementation
def cyl_bessel_j(v, z):
    """Bessel function of the first kind."""
    return mp.besselj(v, z)


@overload
def cyl_bessel_je(v: float, z: float) -> float: ...
@overload
def cyl_bessel_je(v: float, z: complex) -> complex: ...


@reference_implementation
def cyl_bessel_je(v, z):
    """Exponentially scaled Bessel function of the first kind."""
    return mp.exp(-abs(z.imag)) * mp.besselj(v, z)


@reference_implementation
def cyl_bessel_k0(x: float) -> float:
    """Modified Bessel function of the second kinf of order 0."""
    return mp.besselk(0, x)


@reference_implementation
def cyl_bessel_k0e(x: float) -> float:
    """Exponentially scaled modified Bessel function K of order 0."""
    return mp.exp(x) * mp.besselk(0, x)


@reference_implementation
def cyl_bessel_k1(x: float) -> float:
    """Modified Bessel function of the second kind of order 0."""
    return mp.besselk(1, x)


@reference_implementation
def cyl_bessel_k1e(x: float) -> float:
    """Exponentially scaled modified Bessel function K of order 1."""
    return mp.exp(x) * mp.besselk(1, x)


@overload
def cyl_bessel_k(v: float, z: float) -> float: ...
@overload
def cyl_bessel_k(v: float, z: complex) -> complex: ...


@reference_implementation
def cyl_bessel_k(v, z):
    """Modified Bessel function of the second kind."""
    return mp.besselk(v, z)


@overload
def cyl_bessel_ke(v: float, z: float) -> float: ...
@overload
def cyl_bessel_ke(v: float, z: complex) -> complex: ...
    

@reference_implementation
def cyl_bessel_ke(v, z):
    """Exponentially scaled modified Bessel function of the second kind."""
    return mp.exp(z) * mp.besselk(v, z)


@reference_implementation
def cyl_bessel_y0(z: float) -> float:
    """Bessel function of the second kind of order 0."""
    return mp.bessely(0, z)


@reference_implementation
def cyl_bessel_y1(z: float) -> float:
    """Bessel function of the second kind of order 1."""
    return mp.bessely(1, z)


@overload
def cyl_bessel_y(v: float, z: float) -> float: ...
@overload
def cyl_bessel_y(v: float, z: complex) -> complex: ...


@reference_implementation
def cyl_bessel_y(v, z):
    """Bessel function of the second kind."""
    return mp.bessely(v, z)


@overload
def cyl_bessel_ye(v: float, z: float) -> float: ...
@overload
def cyl_bessel_ye(v: float, z: complex) -> complex: ...


@reference_implementation
def cyl_bessel_ye(v, z):
    """Exponentially scaled Bessel function of the second kind."""
    return mp.bessely(v, z) * mp.exp(-abs(z.imag))


@overload
def digamma(x: float) -> float: ...
@overload
def digamma(x: complex) -> complex: ...


@reference_implementation
def digamma(x):
    """The digamma function."""
    return mp.digamma(x)


@reference_implementation
def ellipe(m: float) -> float:
    """Complete elliptic integral of the second kind."""
    return mp.ellipe(m)


@reference_implementation
def ellipeinc(phi: float, m: float) -> float:
    """Incomplete elliptic integral of the second kind."""
    return mp.ellipe(phi, m)


@reference_implementation
def ellipk(m: float) -> float:
    """Complete elliptic integral of the first kind."""
    return mp.ellipk(m)


@reference_implementation
def ellipkm1(p: float) -> float:
    """Complete elliptic integral of the first kind around m = 1."""
    return mp.ellipk(1 - p)


@reference_implementation
def ellipkinc(phi: float, m: float) -> float:
    """Incomplete elliptic integral of the first kind."""
    return mp.ellipf(phi, m)


@reference_implementation
def ellipj(u: float, m: float) -> Tuple[float, float, float, float]:
    """Jacobian Elliptic functions."""
    sn = mp.ellipfun("sn", u=u, m=m)
    cn = mp.ellipfun("cn", u=u, m=m)
    dn = mp.ellipfun("dn", u=u, m=m)
    phi = mp.asin(sn)
    return sn, cn, dn, phi


@reference_implementation
def erf(x: float) -> float:
    """Error function."""
    return mp.erf(x)


@reference_implementation
def erfc(x: float) -> float:
    """Complementary error function 1 - erf(x)."""
    return mp.erfc(x)


@reference_implementation
def erfcinv(x: float) -> float:
    """Inverse of the complementary error function."""
    return mp.erfinv(mp.one - mp.mpf(x))


@overload
def exp1(x: float) -> float: ...
@overload
def exp1(x: complex) -> complex: ...


@reference_implementation
def exp1(x):
    """Exponential integral E1."""
    return mp.e1(x)


@reference_implementation
def exp10(x: float) -> float:
    """Compute 10**x."""
    return mp.mpf(10) ** x


@reference_implementation
def exp2(x: float) -> float:
    """Compute 2**x."""
    return mp.mpf(2) ** x


@overload
def expi(x: float) -> float: ...
@overload
def expi(x: complex) -> complex: ...


@reference_implementation
def expi(x):
    """Exponential integral Ei."""
    return mp.ei(x)


@reference_implementation
def expit(x: float) -> float:
    """Expit (a.k.a logistic sigmoid)."""
    return mp.sigmoid(x)


@reference_implementation
def expm1(x: float) -> float:
    """exp(x) - 1."""
    return mp.expm1(x)


@reference_implementation
def expn(n: int, x: float) -> float:
    """Generalized exponential integral En."""
    return mp.expint(n, x)


@reference_implementation
def exprel(x: float) -> float:
    """Relative error exponential, (exp(x) - 1)/x."""
    return (mp.exp(x) - 1) / x


@reference_implementation
def fdtr(dfn: float, dfd: float, x: float) -> float:
    """F cumulative distribution function."""
    x_dfn = x * dfn
    return mp.betainc(dfn / 2, dfd / 2, 0, x_dfn / (dfd + x_dfn), regularized=True)


@reference_implementation
def fdtrc(dfn: float, dfd: float, x: float) -> float:
    """F survival function."""
    x_dfn = x * dfn
    return mp.betainc(dfn / 2, dfd / 2, x_dfn / (dfd + x_dfn), 1.0, regularized=True)


@overload
def fresnel(x: float) -> Tuple[float, float]: ...
@overload
def fresnel(x: complex) -> Tuple[complex, complex]: ...


@reference_implementation
def fresnel(x):
    """Fresnel integrals."""
    return mp.fresnels(x), mp.fresnelc(x)


@overload
def gamma(x: float) -> float: ...
@overload
def gamma(x: complex) -> complex: ...


@reference_implementation
def gamma(x):
    """Gamma function."""
    if x == 0.0:
        if isinstance(x, float):
            return math.copysign(mp.inf, x)
        return mp.nan
    if x.real < 0 and x.imag == 0 and x.real == int(x.real):
        return mp.nan
    return mp.gamma(x)


@reference_implementation
def gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function."""
    return mp.gammainc(a, 0, x, regularized=True)


@reference_implementation
def gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function."""
    return mp.gammainc(a, x, mp.inf, regularized=True)


@reference_implementation
def gammaln(x: float) -> float:
    """Logarithm of the absolute value of the gamma function."""
    if x.real <= 0 and x == int(x):
        return mp.inf
    return mp.log(abs(mp.gamma(x)))


@reference_implementation
def gammasgn(x: float) -> float:
    """Sign of the gamma function."""
    if x == 0.0:
        return math.copysign(1.0, x)
    if x < 0 and x == int(x):
        return mp.nan
    return mp.sign(mp.gamma(x))


@reference_implementation
def gdtr(a: float, b: float, x: float) -> float:
    """Gamma distribution cumulative distribution function."""
    return mp.gammainc(b, 0, a * x, regularized=True)


@reference_implementation
def gdtrc(a: float, b: float, x: float)-> float:
    """Gamma distribution survival function."""
    return mp.gammainc(b, a * x, mp.inf, regularized=True)


@overload
def hankel1(v: float, z: float) -> float: ...
@overload
def hankel1(v: float, z: complex) -> complex: ...


@reference_implementation
def hankel1(v, z):
    """Hankel function of the first kind."""
    return mp.hankel1(v, z)


@overload
def hankel1e(v: float, z: float) -> float: ...
@overload
def hankel1e(v: float, z: complex) -> complex: ...


@reference_implementation
def hankel1e(v, z):
    """Exponentially scaled Hankel function of the first kind."""
    return mp.hankel1(v, z) * mp.exp(z * -1j)


@overload
def hankel2(v: float, z: float) -> float: ...
@overload
def hankel2(v: float, z: complex) -> complex: ...


@reference_implementation
def hankel2(v, z):
    """Hankel function of the second kind."""
    return mp.hankel2(v, z)


@overload
def hankel2e(v: float, z: float) -> float: ...
@overload
def hankel2e(v: float, z: complex) -> complex: ...


@reference_implementation
def hankel2e(v, z):
    """Exponentially scaled Hankel function of the second kind."""
    return mp.hankel2(v, z) * mp.exp(z * 1j)


@reference_implementation
def hyp1f1(a: float, b: float, z: complex) -> complex:
    """Confluent hypergeometric function 1F1."""
    return mp.hyp1f1(a, b, z)


@overload
def hyp2f1(a: float, b: float, c: float, z: float) -> float: ...
@overload
def hyp2f1(a: float, b: float, c: float, z: complex) -> complex: ...


@reference_implementation
def hyp2f1(a, b, c, z):
    """Gauss hypergeometric function 2F1(a, b; c; z)."""
    if z.imag == 0 and z.real > 1:
        # On branch cut, choose branch based on sign of zero
        z += mp.mpc("0", "1e-1000000") * math.copysign(z.imag)
    return mp.hyp2f1(a, b, c, z)


@reference_implementation
def hyperu(a: float, b: float, z: float) -> float:
    """Confluent hypergeometric function U."""
    return mp.hyperu(a, b, z)


@reference_implementation
def kei(x: float) -> float:
    """Kelvin function kei."""
    return mp.kei(0, x)


@reference_implementation
def ker(x: float) -> float:
    """Kelvin function ker."""
    return mp.ker(0, x)


@reference_implementation
def lambertw(z: complex, k: int) -> complex:
    """Lambert W function."""
    return mp.lambertw(z, k=k)


@reference_implementation
def lanczos_sum_expg_scaled(z: float) -> float:
    """Exponentially scaled Lanczos approximation to the Gamma function."""
    g = mp.mpf("6.024680040776729583740234375")
    return (mp.e / (z + g - 0.5)) ** (z - 0.5) * mp.gamma(z)


@reference_implementation
def lgam1p(x: float) -> float:
    """Logarithm of gamma(x + 1)."""
    return mp.log(mp.gamma(x + 1))


@reference_implementation
def log1p(x: float) -> float:
    """Logarithm of x + 1."""
    return mp.log1p(x)


@reference_implementation
def log1pmx(x: float) -> float:
    """log(x + 1) - x."""
    return mp.log1p(x) - x


@overload
def loggamma(z: float) -> float: ...
@overload
def loggamma(z: complex) -> complex: ...


@reference_implementation
def loggamma(z):
    """Principal branch of the logarithm of the gamma function."""
    return mp.loggamma(z)


@reference_implementation
def log_wright_bessel(a: float, b: float, x: float) -> float:
    """Natural logarithm of Wright's generalized Bessel function."""
    return mp.log(_wright_bessel(a, b, x))


@reference_implementation
def nbdtr(k: int, n: int, p: float) -> float:
    """Negative binomial cumulative distribution function."""
    return mp.betainc(n, k + 1, 0, p)


@reference_implementation
def nbdtrc(k: int, n: int, p: float) -> float:
    """Negative binomial survival function."""
    return mp.betainc(n, k + 1, p, 1)


@reference_implementation
def ndtr(x: float) -> float:
    """Cumulative distribution of the standard normal distribution."""
    return mp.ncdf(x)


@reference_implementation
def owens_t(h: float, a: float) -> float:
    """Owen's T Function."""
    def integrand(x):
        return mp.exp(-(h**2) * (1 + x**2) / 2) / (1 + x**2)

    return mp.quad(integrand, [0, a]) / (2 * mp.pi)


@reference_implementation
def pdtr(k: float, m: float) -> float:
    """Poisson cumulative distribution function."""
    k =mp.floor(k)

    def term(j):
        return m**j / mp.factorial(j)

    return mp.exp(-m) * mp.nsum(term, [0, k])


@reference_implementation
def pdtrc(k: float, m: float) -> float:
    """Poisson survival function."""
    k = mp.floor(k)
    return mp.gammainc(k + 1, 0, m, regularized=True)


@reference_implementation
def poch(z: float, m: float) -> float:
    """Pochhammer symbol."""
    return mp.rf(z, m)


@overload
def rgamma(z: float) -> float: ...
@overload
def rgamma(z: complex) -> complex: ...


@reference_implementation
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


@reference_implementation
def riemann_zeta(z):
    """Riemann zeta function."""
    if z == 1.0:
        return mp.nan
    return mp.zeta(z)


@reference_implementation
def scaled_exp1(x: float) -> float:
    """Exponentially scaled exponential integral E1."""
    return mp.exp(x) * x * mp.e1(x)


@overload
def shichi(x: float) -> Tuple[float, float]: ...
@overload
def shichi(x: complex) -> Tuple[complex, complex]: ...


@reference_implementation
def shichi(x):
    """Hyperbolic sine and cosine integrals."""
    return mp.shi(x), mp.chi(x)


@overload
def sici(x: float) -> Tuple[float, float]: ...
@overload
def sici(x: complex) -> Tuple[complex, complex]: ...


@reference_implementation
def sici(x):
    """Sine and cosine integrals."""
    return mp.si(x), mp.ci(x)


@overload
def sinpi(x: float) -> float: ...
@overload
def sinpi(x: complex) -> complex: ...


@reference_implementation
def sinpi(x):
    """Sine of pi*x."""
    return mp.sinpi(x)


@reference_implementation
def spence(z: float) -> float:
    """Spence's function, also known as the dilogarithm."""
    return mp.polylog(2, 1 - z)


@reference_implementation
def struve_h(v: float, x: float) -> float:
    """Struve function."""
    return mp.struveh(v, x)


@reference_implementation
def struve_l(v: float, x: float) -> float:
    """Modified Struve function."""
    return  mp.struvel(v, x)


@reference_implementation
def tandg(x: float) -> float:
    """Tangent of angle x given in degrees."""
    return mp.tan(mp.radians(x))


@reference_implementation
def wright_bessel(a: float, b: float, x: float) -> float:
    """Wright's generalized Bessel function."""
    return _wright_bessel(a, b, x)


@reference_implementation
def zeta(z: float, q: float) -> float:
    """Hurwitz zeta function."""
    if z == 1.0:
        return mp.nan
    return mp.zeta(z, a=q)


@reference_implementation
def zetac(z: float) -> float:
    """Riemann zeta function minus 1."""
    if z == 1.0:
        return mp.nan
    return mp.zeta(z) - mp.one


def _wright_bessel(a, b, x):
    a, b, x = (_to_arbitrary_precision(t) for t in (a, b, x))

    def term(k):
        return x**k / (mp.factorial(k) * mp.gamma(a * k + b))

    return mp.nsum(term, [0, mp.inf])


_exclude = ["math", "mp", "overload", "reference_implementation", "Tuple"]

__all__ = [s for s in dir() if not s.startswith("_") and s not in _exclude]
