import math

from mpmath import mp  # type: ignore
from typing import overload, Tuple


@overload
def airy(x: float) -> Tuple[float, float, float, float]: ...
@overload
def airy(x: complex) -> Tuple[complex, complex, complex, complex]: ...


def airy(z):
    """Airy functions and their derivatives."""
    ai = mp.airyai(z)
    aip = mp.airyai(z, derivative=1)
    bi = mp.airybi(z)
    bip = mp.airybi(z, derivative=1)
    return tuple(_to_finite_precision(t) for t in  (ai, aip, bi, bip))


@overload
def airye(x: float) -> Tuple[float, float, float, float]: ...
@overload
def airye(x: complex) -> Tuple[complex, complex, complex, complex]: ...


def airye(z):
    """Exponentially scaled Airy functions and their derivatives."""
    z = _to_arbitrary_precision(z)
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
    return tuple(_to_finite_precision(t) for t in (eai, eaip, ebi, ebip))


def bei(x: float) -> float:
    """Kelvin function bei."""
    result = mp.bei(0, x)
    return _to_finite_precision(result)


def ber(x: float) -> float:
    """Kelvin function ber."""
    result = mp.ber(0, x)
    return _to_finite_precision(result)


def besselpoly(a: float, lmb: float, nu: float) -> float:
    """Weighted integral of the Bessel function of the first kind."""
    a, lmb, nu = (_to_arbitrary_precision(t) for t in (a, lmb, nu))

    def integrand(x):
        return x**lmb * mp.besselj(nu, 2 * a * x)

    result = mp.quad(integrand, [0, 1])
    return _to_finite_precision(result)


def beta(a: float, b: float) -> float:
    """Beta function."""
    result = mp.beta(a, b)
    return _to_finite_precision(result)


def betaln(a: float, b: float) -> float:
    """Natural logarithm of the absolute value of the Beta function."""
    result = mp.log(abs(mp.beta(a, b)))
    return _to_finite_precision(result)


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete Beta function."""
    result = mp.betainc(a, b, 0, x, regularized=True)
    return _to_finite_precision(result)


def betaincc(a, b, x):
    """Complement of the regularized incomplete Beta function."""
    result = mp.betainc(a, b, x, 1.0, regularized=True)
    return _to_finite_precision(result)


def bdtr(k: float, n: float, p: float) -> float:
    """Binomial distribution cumulative distribution function."""
    k, n, p = mp.floor(k), mp.floor(n), mp.mpf(p)
    result = float(mp.betainc(n - k, k + 1, 0, 1 - p, regularized=True))
    return result


def bdtrc(k: float, n: float, p: float) -> float:
    """Binomial distribution survival function."""
    k, n, p = mp.floor(k), mp.floor(n), mp.mpf(p)
    result = float(mp.betainc(k + 1, n - k, 0, p, regularized=True))
    return result


def binom(n: float, k: float) -> float:
    """Binomial coefficient considered as a function of two real variables."""
    result = mp.binomial(n, k)
    return _to_finite_precision(result)


def cbrt(x: float) -> float:
    """Cube root of x."""
    result = mp.cbrt(x)
    return _to_finite_precision(result)


def chdtr(v: float, x: float) -> float:
    """Chi square cumulative distribution function."""
    v, x = (_to_arbitrary_precision(t) for t in (v, x))
    result = mp.gammainc(v / 2, 0, x / 2, regularized=True)
    return _to_finite_precision(result)


def chdtrc(v: float, x: float) -> float:
    """Chi square survival function."""
    v, x = (_to_arbitrary_precision(t) for t in (v, x))
    result = mp.gammainc(v / 2, x / 2, mp.inf, regularized=True)
    return _to_finite_precision(result)


def cosdg(x: float) -> float:
    """Cosine of the angle x given in degrees."""
    result = mp.cos(mp.radians(x))
    return _to_finite_precision(result)


def cosm1(x: float) -> float:
    """cos(x) - 1 for use when x is near zero."""
    result = mp.cos(x) - 1
    return _to_finite_precision(result)


@overload
def cospi(x: float) -> float: ...
@overload
def cospi(x: complex) -> complex: ...


def cospi(x):
    """Cosine of pi*x."""
    result = mp.cospi(x)
    return _to_finite_precision(x)


def cotdg(x: float) -> float:
    """Cotangent of the angle x given in degrees."""
    result = mp.cot(mp.radians(x))
    return _to_finite_precision(result)


def cyl_bessel_i0(z: float) -> float:
    """Modified Bessel function of order 0."""
    result = mp.besseli(0, z)
    return _to_finite_precision(result)


def cyl_bessel_i0e(z: float) -> float:
    """Exponentially scaled modified Bessel function of order 0."""
    z = _to_arbitrary_precision(z)
    result = mp.exp(-abs(z.real)) * mp.besseli(0, z)
    return _to_finite_precision(result)


def cyl_bessel_i1(z: float)-> float:
    """Modified Bessel function of order 1."""
    result = mp.besseli(1, z)
    return _to_finite_precision(result)


def cyl_bessel_i1e(z: float) -> float:
    """Exponentially scaled modified Bessel function of order 1."""
    z = _to_arbitrary_precision(z)
    result = mp.exp(-abs(z.real)) * mp.besseli(1, z)
    return _to_finite_precision(result)


@overload
def cyl_bessel_i(v: float, z: float) -> float: ...
@overload
def cyl_bessel_i(v: float, z:complex) -> complex: ...


def cyl_bessel_i(v, z):
    """Modified Bessel function of the first kind."""
    result = mp.besseli(v, z)
    return _to_finite_precision(result)


@overload
def cyl_bessel_ie(v: float, z: float) -> float: ...
@overload
def cyl_bessel_ie(v: float, z:complex) -> complex: ...



def cyl_bessel_ie(v, x):
    """Exponentially scaled modified Bessel function of the first kind."""
    z = _to_arbitrary_precision(z)
    result = mp.exp(-abs(z.real)) * mp.besseli(v, z)
    return _to_finite_precision(result)


def cyl_bessel_j0(x: float) -> float:
    """Bessel function of the first kind of order 0."""
    result = mp.j0(x)
    return _to_finite_precision(result)


def cyl_bessel_j1(x: float) -> float:
    """Bessel function of the first kind of order 1."""
    result = mp.j1(x)
    return _to_finite_precision(result)


@overload
def cyl_bessel_j(v: float, z: float) -> float: ...
@overload
def cyl_bessel_j(v: float, z: complex) -> complex: ...


def cyl_bessel_j(v, z):
    """Bessel function of the first kind."""
    result = mp.besselj(v, z)
    return _to_finite_precision(result)


@overload
def cyl_bessel_je(v: float, z: float) -> float: ...
@overload
def cyl_bessel_je(v: float, z: complex) -> complex: ...


def cyl_bessel_je(v, z):
    """Exponentially scaled Bessel function of the first kind."""
    z = _to_arbitrary_precision(z)
    result = mp.exp(-abs(z.imag)) * mp.besselj(v, z)


def cyl_bessel_k0(x: float) -> float:
    """Modified Bessel function of the second kinf of order 0."""
    result = mp.besselk(0, x)
    return _to_finite_precision(result)


def cyl_bessel_k0e(x: float) -> float:
    """Exponentially scaled modified Bessel function K of order 0."""
    result = mp.exp(x) * mp.besselk(0, x)
    return _to_finite_precision(result)


def cyl_bessel_k1(x: float) -> float:
    """Modified Bessel function of the second kind of order 0."""
    result = mp.besselk(1, x)
    return _to_finite_precision(x)


def cyl_bessel_k1e(x: float) -> float:
    """Exponentially scaled modified Bessel function K of order 1."""
    result = mp.exp(x) * mp.besselk(1, x)
    return _to_finite_precision(result)


@overload
def cyl_bessel_k(v: float, z: float) -> float: ...
@overload
def cyl_bessel_k(v: float, z: complex) -> complex: ...


def cyl_bessel_k(v, z):
    """Modified Bessel function of the second kind."""
    result = mp.besselk(v, z)
    return _to_finite_precision(result)


@overload
def cyl_bessel_ke(v: float, z: float) -> float: ...
@overload
def cyl_bessel_ke(v: float, z: complex) -> complex: ...
    

def cyl_bessel_ke(v, z):
    """Exponentially scaled modified Bessel function of the second kind."""
    result = mp.exp(z) * mp.besselk(v, z)
    return _to_finite_precision(result)


def cyl_bessel_y0(z: float) -> float:
    """Bessel function of the second kind of order 0."""
    result = mp.bessely(0, z)
    return _to_finite_precision(result)


def cyl_bessel_y1(z: float) -> float:
    """Bessel function of the second kind of order 1."""
    result = mp.bessely(1, z)
    return _to_finite_precision(result)


@overload
def cyl_bessel_y(v: float, z: float) -> float: ...
@overload
def cyl_bessel_y(v: float, z: complex) -> complex: ...


def cyl_bessel_y(v, z):
    """Bessel function of the second kind."""
    result = mp.bessely(v, z)
    return _to_finite_precision(result)


@overload
def cyl_bessel_ye(v: float, z: float) -> float: ...
@overload
def cyl_bessel_ye(v: float, z: complex) -> complex: ...


def cyl_bessel_ye(v, z):
    """Exponentially scaled Bessel function of the second kind."""
    z = _to_arbitrary_precision(z)
    result = mp.bessely(v, z) * mp.exp(-abs(z.imag))
    return _to_finite_precision(result)


@overload
def digamma(x: float) -> float: ...
@overload
def digamma(x: complex) -> complex: ...


def digamma(x):
    """The digamma function."""
    result = mp.digamma(x)
    return _to_finite_precision(result)


def ellipe(m: float) -> float:
    """Complete elliptic integral of the second kind."""
    result = mp.ellipe(m)
    return _to_finite_precision(result)


def ellipeinc(phi: float, m: float) -> float:
    """Incomplete elliptic integral of the second kind."""
    result = mp.ellipe(phi, m)
    return _to_finite_precision(result)


def ellipk(m: float) -> float:
    """Complete elliptic integral of the first kind."""
    result = mp.ellipk(m)
    return _to_finite_precision(result)


def ellipkm1(p: float) -> float:
    """Complete elliptic integral of the first kind around m = 1."""
    p = _to_arbitrary_precision(p)
    result = mp.ellipk(1 - p)
    return _to_finite_precision(result)


def ellipkinc(phi: float, m: float) -> float:
    """Incomplete elliptic integral of the first kind."""
    result = mp.ellipf(phi, m)
    return _to_finite_precision(result)


def ellipj(u: float, m: float) -> Tuple[float, float, float, float]:
    """Jacobian Elliptic functions."""
    sn = mp.ellipfun("sn", u=u, m=m)
    cn = mp.ellipfun("cn", u=u, m=m)
    dn = mp.ellipfun("dn", u=u, m=m)
    phi = mp.asin(sn)
    return tuple(_to_finite_precision(t) for t in (sn, cn, dn, phi))


def erf(x: float) -> float:
    """Error function."""
    result = mp.erf(x)
    return _to_finite_precision(result)


def erfc(x: float) -> float:
    """Complementary error function 1 - erf(x)."""
    result = mp.erfc(x)
    return _to_finite_precision(result)


def erfcinv(x: float) -> float:
    """Inverse of the complementary error function."""
    result = mp.erfinv(mp.one - mp.mpf(x))
    return _to_finite_precision(result)


@overload
def exp1(x: float) -> float: ...
@overload
def exp1(x: complex) -> complex: ...


def exp1(x):
    """Exponential integral E1."""
    result = mp.e1(x)
    return _to_finite_precision(result)


def exp10(x: float) -> float:
    """Compute 10**x."""
    result = mp.mpf(10) ** x
    return _to_finite_precision(result)


def exp2(x: float) -> float:
    """Compute 2**x."""
    result = mp.mpf(2) ** x
    return _to_finite_precision(x)


@overload
def expi(x: float) -> float: ...
@overload
def expi(x: complex) -> complex: ...


def expi(x):
    """Exponential integral Ei."""
    result = mp.ei(x)
    return _to_finite_precision(result)


def expit(x: float) -> float:
    """Expit (a.k.a logistic sigmoid)."""
    result = mp.sigmoid(x)
    return _to_finite_precision(x)


def expm1(x: float) -> float:
    """exp(x) - 1."""
    result = mp.expm1(x)
    return _to_finite_precision(result)


def expn(n: int, x: float) -> float:
    """Generalized exponential integral En."""
    result = mp.expint(n, x)
    return _to_finite_precision(result)


def exprel(x: float) -> float:
    """Relative error exponential, (exp(x) - 1)/x."""
    x = _to_arbitrary_precision(x)
    result = (mp.exp(x) - 1) / x
    return _to_finite_precision(x)


def fdtr(dfn: float, dfd: float, x: float) -> float:
    """F cumulative distribution function."""
    dfn, dfd, x = (_to_arbitrary_precision(t) for t in (dfn, dfd, x))
    x_dfn = x * dfn
    result = mp.betainc(dfn / 2, dfd / 2, 0, x_dfn / (dfd + x_dfn), regularized=True)
    return _to_finite_precision(result)


def fdtrc(dfn: float, dfd: float, x: float) -> float:
    """F survival function."""
    dfn, dfd, x = (_to_arbitrary_precision(t) for t in (dfn, dfd, x))
    x_dfn = x * dfn
    result = mp.betainc(dfn / 2, dfd / 2, x_dfn / (dfd + x_dfn), 1.0, regularized=True)
    return _to_finite_precision(result)


@overload
def fresnel(x: float) -> Tuple[float, float]: ...
@overload
def fresnel(x: complex) -> Tuple[complex, complex]: ...


def fresnel(x):
    """Fresnel integrals."""
    S, C = mp.fresnels(x), mp.fresnelc(x)
    return tuple(_to_finite_precision(t) for t in (S, C))


@overload
def gamma(x: float) -> float: ...
@overload
def gamma(x: complex) -> complex: ...


def gamma(x):
    """Gamma function."""
    if x == 0.0:
        return math.copysign(math.inf, x)
    if x < 0 and x == int(x):
        return math.nan
    result = mp.gamma(x)
    return _to_finite_precision(result)


def gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function."""
    result = mp.gammainc(a, 0, x, regularized=True)
    return _to_finite_precision(result)


def gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function."""
    result = mp.gammainc(a, x, mp.inf, regularized=True)
    return _to_finite_precision(result)


def gammaln(x: float) -> float:
    """Logarithm of the absolute value of the gamma function."""
    result = mp.log(abs(mp.gamma(x)))
    return _to_finite_precision(result)


def gammasgn(x: float) -> float:
    """Sign of the gamma function."""
    if x == 0.0:
        return math.copysign(1.0, x)
    if x < 0 and x == int(x):
        return math.nan
    return mp.sign(mp.gamma(x))


def gdtr(a: float, b: float, x: float) -> float:
    """Gamma distribution cumulative distribution function."""
    a, b, x = (_to_arbitrary_precision(t) for t in (a, b, x))
    result = mp.gammainc(b, 0, a * x, regularized=True)
    return _to_finite_precision(result)


def gdtrc(a: float, b: float, x: float)-> float:
    """Gamma distribution survival function."""
    a, b, x = (_to_arbitrary_precision(t) for t in (a, b, x))
    result = mp.gammainc(b, a * x, mp.inf, regularized=True)
    return _to_finite_precision(result)


@overload
def hankel1(v: float, z: float) -> float: ...
@overload
def hankel1(v: float, z: complex) -> complex: ...


def hankel1(v, z):
    """Hankel function of the first kind."""
    result = mp.hankel1(v, z)
    return _to_finite_precision(result)


@overload
def hankel1e(v: float, z: float) -> float: ...
@overload
def hankel1e(v: float, z: complex) -> complex: ...


def hankel1e(v, z):
    """Exponentially scaled Hankel function of the first kind."""
    v, z = (_to_arbitrary_precision(t) for t in (v, z))
    result = mp.hankel1(v, z) * mp.exp(z * -1j)
    return _to_finite_precision(result)


@overload
def hankel2(v: float, z: float) -> float: ...
@overload
def hankel2(v: float, z: complex) -> complex: ...


def hankel2(v, z):
    """Hankel function of the second kind."""
    result = mp.hankel2(v, z)
    return _to_finite_precision(result)


@overload
def hankel2e(v: float, z: float) -> float: ...
@overload
def hankel2e(v: float, z: complex) -> complex: ...


def hankel2e(v, z):
    """Exponentially scaled Hankel function of the second kind."""
    v, z = _to_arbitrary_precision(v, z)
    result = mp.hankel2(v, z) * mp.exp(z * 1j)
    return _to_finite_precision(result)


def hyp1f1(a: float, b: float, z: complex) -> complex:
    """Confluent hypergeometric function 1F1."""
    result = mp.hyp1f1(a, b, z)
    return _to_finite_precision(result)


@overload
def hyp2f1(a: float, b: float, c: float, z: float) -> float: ...
@overload
def hyp2f1(a: float, b: float, c: float, z: complex) -> complex: ...


def hyp2f1(a, b, c, z):
    """Gauss hypergeometric function 2F1(a, b; c; z)."""
    result = mp.hyp2f1(a, b, c, z)
    return _to_finite_precision(result)


def hyperu(a: float, b: float, z: float) -> float:
    """Confluent hypergeometric function U."""
    result = mp.hyperu(a, b, z)
    return _to_finite_precision(result)


def kei(x: float) -> float:
    """Kelvin function kei."""
    result = mp.kei(0, x)
    return _to_finite_precision(result)


def ker(x: float) -> float:
    """Kelvin function ker."""
    result = mp.ker(0, x)
    return _to_finite_precision(result)


def lambertw(z: complex, k: int) -> complex:
    """Lambert W function."""
    result = mp.lambertw(z, k=k)
    return _to_finite_precision(result)


def lanczos_sum_expg_scaled(z: float) -> float:
    """Exponentially scaled Lanczos approximation to the Gamma function."""
    z = _to_arbitrary_precision(z)
    g = mp.mpf("6.024680040776729583740234375")
    result = (mp.e / (z + g - 0.5)) ** (z - 0.5) * mp.gamma(z)
    return _to_finite_precision(result)


def lgam1p(x: float) -> float:
    """Logarithm of gamma(x + 1)."""
    x = _to_arbitrary_precision(x)
    result = mp.log(mp.gamma(x + 1))
    return _to_finite_precision(result)


def log1p(x: float) -> float:
    """Logarithm of x + 1."""
    result = mp.log1p(x)
    return _to_finite_precision(x)


def log1pmx(x: float) -> float:
    """log(x + 1) - x."""
    x = _to_arbitrary_precision(x)
    result = mp.log1p(x) - x
    return _to_finite_precision(result)


@overload
def loggamma(z: float) -> float: ...
@overload
def loggamma(z: complex) -> complex: ...


def loggamma(z):
    """Principal branch of the logarithm of the gamma function."""
    result = mp.loggamma(z)
    return _to_finite_precision(result)


def log_wright_bessel(a: float, b: float, x: float) -> float:
    """Natural logarithm of Wright's generalized Bessel function."""
    result = mp.log(_wright_bessel(a, b, x))
    return _to_finite_precision(result)


def nbdtr(k: int, n: int, p: float) -> float:
    """Negative binomial cumulative distribution function."""
    k, n, p = (_to_arbitrary_precision(t) for t in (k, n, p))
    result = mp.betainc(n, k + 1, 0, p)
    return _to_finite_precision(result)


def nbdtrc(k: int, n: int, p: float) -> float:
    """Negative binomial survival function."""
    k, n, p = (_to_arbitrary_precision(t) for t in (k, n, p))
    result = mp.betainc(n, k + 1, p, 1)
    return _to_finite_precision(result)


def ndtr(x: float) -> float:
    """Cumulative distribution of the standard normal distribution."""
    result = mp.ncdf(x)
    return _to_finite_precision(x)


def owens_t(h: float, a: float) -> float:
    """Owen's T Function."""
    h, a = (_to_arbitrary_precision(t) for t in (h, a))

    def integrand(x):
        return mp.exp(-(h**2) * (1 + x**2) / 2) / (1 + x**2)

    result = mp.quad(integrand, [0, a]) / (2 * mp.pi)
    return _to_finite_precision(result)


def pdtr(k: float, m: float) -> float:
    """Poisson cumulative distribution function."""
    k, m = mp.floor(k), mp.mpf(m)

    def term(j):
        return m**j / mp.factorial(j)

    result = mp.exp(-m) * mp.nsum(term, [0, k])
    return _to_finite_precision(result)


def pdtrc(k: float, m: float) -> float:
    """Poisson survival function."""
    k, m = mp.floor(k), mp.mpf(m)
    result = mp.gammainc(k + 1, 0, m, regularized=True)
    return _to_finite_precision(result)


def poch(z: float, m: float) -> float:
    """Pochhammer symbol."""
    result = mp.rf(z, m)
    return _to_finite_precision(result)


@overload
def rgamma(z: float) -> float: ...
@overload
def rgamma(z: complex) -> complex: ...


def rgamma(z):
    """Reciprocal of the gamma function."""
    if x == 0.0:
        return x
    if x < 0 and x == int(x):
        return 0.0
    result = mp.one / mp.gamma(z)
    return _to_finite_precision(result)


@overload
def riemann_zeta(z: float) -> float: ...
@overload
def riemann_zeta(z: complex) -> complex: ...


def riemann_zeta(z):
    """Riemann zeta function."""
    if z == 1.0:
        return math.nan
    result = mp.zeta(z)
    return _to_finite_precision(result)


def scaled_exp1(x: float) -> float:
    """Exponentially scaled exponential integral E1."""
    x = _to_arbitrary_precision(x)
    result = mp.exp(x) * x * mp.e1(x)
    return _to_finite_precision(result)


@overload
def shichi(x: float) -> Tuple[float, float]: ...
@overload
def shichi(x: complex) -> Tuple[complex, complex]: ...


def shichi(x):
    """Hyperbolic sine and cosine integrals."""
    shi, chi = mp.shi(x), mp.chi(x)
    return _to_finite_precision(shi), _to_finite_precision(chi)


@overload
def sici(x: float) -> Tuple[float, float]: ...
@overload
def sici(x: complex) -> Tuple[complex, complex]: ...


def sici(x):
    """Sine and cosine integrals."""
    si, ci = mp.si(x), mp.ci(x)
    return _to_finite_precision(si), _to_finite_precision(ci)


@overload
def sinpi(x: float) -> float: ...
@overload
def sinpi(x: complex) -> complex: ...


def sinpi(x):
    """Sine of pi*x."""
    result = mp.sinpi(x)
    return _to_finite_precision(result)


def spence(z: float) -> float:
    """Spence's function, also known as the dilogarithm."""
    result = mp.polylog(2, 1 - z)
    return _to_finite_precision(result)


def struve_h(v: float, x: float) -> float:
    """Struve function."""
    result = mp.struveh(v, x)
    return _to_finite_precision(result)


def struve_l(v: float, x: float) -> float:
    """Modified Struve function."""
    result = mp.struvel(v, x)
    return _to_finite_precision(result)


def tandg(x: float) -> float:
    """Tangent of angle x given in degrees."""
    result = mp.tan(mp.radians(x))
    return _to_finite_precision(result)


def wright_bessel(a: float, b: float, x: float) -> float:
    """Wright's generalized Bessel function."""
    result = _wright_bessel(a, b, x)
    return _to_finite_precision(result)


def zeta(z: float, q: float) -> float:
    """Hurwitz zeta function."""
    if z == 1.0:
        return math.nan
    result = mp.zeta(z, a=q)
    return _to_finite_precision(result)


def zetac(z: float) -> float:
    """Riemann zeta function minus 1."""
    if z == 1.0:
        return math.nan
    result = mp.zeta(z) - mp.one
    return _to_finite_precision(result)


def _wright_bessel(a, b, x):
    a, b, x = (_to_arbitrary_precision(t) for t in (a, b, x))

    def term(k):
        return x**k / (mp.factorial(k) * mp.gamma(a * k + b))

    return mp.nsum(term, [0, mp.inf])


def _to_arbitrary_precision(arg):
    """Convert finite precision arguments to arbitrary precison."""
    return mp.mpc(x) if isinstance(x, complex) else mp.mpf(x)


def _to_finite_precision(arg):
    """Convert arbitrary precision arguments to finite precision."""
    return complex(x) if isinstance(x, mp.mpc) else float(x)
