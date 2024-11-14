import math

from mpmath import mp


def _to_arbitrary_precision(*args):
    out = tuple(
        mp.mpc(x) if isinstance(x, complex) else mp.mpf(x) for x in args
    )
    if len(out) == 1:
        out = out[0]
    return out


def _to_finite_precision(*args):
    """Convert mp type to finite precision type."""
    out = tuple(
        complex(x) if isinstance(x, mp.mpc) else float(x) for x in args
    )
    if len(out) == 1:
        out = out[0]
    return out


def airy(z):
    """Airy functions and their derivatives."""
    ai = mp.airyai(z)
    aip = mp.airyai(z, derivative=1)
    bi = mp.airybi(z)
    bip = mp.airybi(z, derivative=1)
    return _to_finite_precision(ai, aip, bi, bip)


def airye(z):
    """Exponentially scaled Airy functions and their derivatives."""
    z = _to_arbitrary_precision(z)
    eai = mp.airyai(z) * mp.exp(mp.mpf("2.0")/mp.mpf("3.0")*z*mp.sqrt(z))
    eaip = (
        mp.airyai(z, derivative=1)
        * mp.exp(mp.mpf("2.0")/mp.mpf("3.0")*z*mp.sqrt(z))
    )
    ebi = (
        mp.airybi(z)
        * mp.exp(-abs(mp.mpf("2.0")/mp.mpf("3.0")*(z*mp.sqrt(z)).real))
    )
    ebip = (
        mp.airybi(z, derivative=1)
        * mp.exp(-abs(mp.mpf("2.0")/mp.mpf("3.0")*(z*mp.sqrt(z)).real))
    )
    return _to_finite_precision(eai, eaip, ebi, ebip)


def bei(x):
    """Kelvin function bei."""
    result = mp.bei(0, x)
    return _to_finite_precision(result)


def ber(x):
    """Kelvin function ber."""
    result = mp.ber(0, x)
    return _to_finite_precision(result)


def besselpoly(a, lmb, nu):
    """Weighted integral of the Bessel function of the first kind."""
    a, lmb, nu = _to_arbitrary_precision(a, lmb, nu)

    def integrand(x):
        return x**lmb * mp.besselj(nu, 2*a*x)

    result = mp.quad(integrand, [0, 1])
    return _to_finite_precision(result)


def beta(a, b):
    """Beta function."""
    result = mp.beta(a, b)
    return _to_finite_precision(result)


def betaln(a, b):
    """Natural logarithm of the absolute value of the Beta function."""
    result = mp.log(abs(mp.beta(a, b)))
    return _to_finite_precision(result)


def betainc(a, b, x):
    """Regularized incomplete Beta function."""
    result = mp.betainc(a, b, 0, x, regularized=True)
    return _to_finite_precision(result)


def betaincc(a, b, x):
    """Complement of the regularized incomplete Beta function."""
    result = mp.betainc(a, b, x, 1.0, regularized=True)
    return _to_finite_precision(result)


def bdtr(k, n, p):
    """Binomial distribution cumulative distribution function."""
    k, n, p = mp.floor(k), mp.floor(n), mp.mpf(p)
    result = float(mp.betainc(n - k, k + 1, 0, 1 - p, regularized=True))
    return result


def bdtrc(k, n, p):
    """Binomial distribution survival function."""
    k, n, p = mp.floor(k), mp.floor(n), mp.mpf(p)
    result = float(mp.betainc(k + 1, n - k, 0, p, regularized=True))
    return result


def binom(n, k):
    """Binomial coefficient considered as a function of two real variables."""
    result = mp.binomial(n, k)
    return _to_finite_precision(result)


def cbrt(x):
    """Cube root of x."""
    result = mp.cbrt(x)
    return _to_finite_precision(result)


def chdtr(v, x):
    """Chi square cumulative distribution function."""
    v, x = _to_arbitrary_precision(v, x)
    result = mp.gammainc(v/2, 0, x/2, regularized=True)
    return _to_finite_precision(result)


def chdtrc(v, x):
    """Chi square survival function."""
    v, x = _to_arbitrary_precision(v, x)
    result = mp.gammainc(v/2, x/2, mp.inf, regularized=True)
    return _to_finite_precision(result)


def cosdg(x):
    """Cosine of the angle x given in degrees."""
    result = mp.cos(mp.radians(x))
    return _to_finite_precision(result)


def cosm1(x):
    """cos(x) - 1 for use when x is near zero."""
    result = mp.cos(x) - 1
    return _to_finite_precision(result)


def cospi(x):
    """Cosine of pi*x."""
    result = mp.cospi(x)
    return _to_finite_precision(x)


def cotdg(x):
    """Cotangent of the angle x given in degrees."""
    result = mp.cot(mp.radians(x))
    return _to_finite_precision(result)


def cyl_bessel_i0(z):
    """Modified Bessel function of order 0."""
    result = mp.besseli(0, z)
    return _to_finite_precision(result)


def cyl_bessel_i0e(z):
    """Exponentially scaled modified Bessel function of order 0."""
    z = _to_arbitrary_precision(z)
    result = mp.exp(-abs(z.real)) * mp.besseli(0, z)
    return _to_finite_precision(result)


def cyl_bessel_i1(z):
    """Modified Bessel function of order 1."""
    result = mp.besseli(1, z)
    return _to_finite_precision(result)


def cyl_bessel_i1e(z):
    """Exponentially scaled modified Bessel function of order 1."""
    z = _to_arbitrary_precision(z)
    result = mp.exp(-abs(z.real)) * mp.besseli(1, z)
    return _to_finite_precision(result)


def cyl_bessel_i(v, z):
    """Modified Bessel function of the first kind."""
    result = mp.besseli(v, z)
    return _to_finite_precision(result)


def cyl_bessel_ie(v, z):
    """Exponentially scaled modified Bessel function of the first kind."""
    z = _to_arbitrary_precision(z)
    result = mp.exp(-abs(z.real)) * mp.besseli(v, z)
    return _to_finite_precision(result)


def cyl_bessel_j0(x):
    """Bessel function of the first kind of order 0."""
    result = mp.j0(x)
    return _to_finite_precision(result)


def cyl_bessel_j1(x):
    """Bessel function of the first kind of order 1."""
    result = mp.j1(x)
    return _to_finite_precision(result)


def cyl_bessel_j(v, z):
    """Bessel function of the first kind."""
    result = mp.besselj(v, z)
    return _to_finite_precision(result)


def cyl_bessel_je(v, z):
    """Exponentially scaled Bessel function of the first kind."""
    z = _to_arbitrary_precision(z)
    result = mp.exp(-abs(z.imag)) * mp.besselj(v, z)


def cyl_bessel_k0(x):
    """Modified Bessel function of the second kinf of order 0."""
    result = mp.besselk(0, x)
    return _to_finite_precision(result)


def cyl_bessel_k0e(x):
    """Exponentially scaled modified Bessel function K of order 0."""
    result = mp.exp(x) * mp.besselk(0, x)
    return _to_finite_precision(result)


def cyl_bessel_k1(x):
    """Modified Bessel function of the second kind of order 0."""
    result = mp.besselk(1, x)
    return _to_finite_precision(x)


def cyl_bessel_k1e(x):
    """Exponentially scaled modified Bessel function K of order 1."""
    result = mp.exp(x) * mp.besselk(1, x)
    return _to_finite_precision(result)


def cyl_bessel_k(v, z):
    """Modified Bessel function of the second kind."""
    result = mp.besselk(v, z)
    return _to_finite_precision(result)


def cyl_bessel_ke(v, z):
    """Exponentially scaled modified Bessel function of the second kind."""
    result = mp.exp(z) * mp.besselk(v, z)
    return _to_finite_precision(result)


def cyl_bessel_y0(z):
    """Bessel function of the second kind of order 0."""
    result = mp.bessely(0, z)
    return _to_finite_precision(v, z)


def cyl_bessel_y1(z):
    """Bessel function of the second kind of order 1."""
    result = mp.bessely(1, z)
    return _to_finite_precision(v, z)


def cyl_bessel_y(v, z):
    """Bessel function of the second kind."""
    result = mp.bessely(v, z)
    return _to_finite_precision(v, z)


def cyl_bessel_ye(v, z):
    """Exponentially scaled Bessel function of the second kind."""
    z = _to_arbitrary_precision(z)
    result = mp.bessely(v, z) * mp.exp(-abs(z.imag))


def digamma(x):
    """The digamma function."""
    result = mp.digamma(x)
    return _to_finite_precision(result)


def ellipe(m):
    """Complete elliptic integral of the second kind."""
    result = mp.ellipe(m)
    return _to_finite_precision(result)


def ellipeinc(phi, m):
    """Incomplete elliptic integral of the second kind."""
    result = mp.ellipe(phi, m)
    return _to_finite_precision(result)


def ellipk(m):
    """Complete elliptic integral of the first kind."""
    result = mp.ellipk(m)
    return _to_finite_precision(result)


def ellipkm1(p):
    """Complete elliptic integral of the first kind around m = 1."""
    p = _to_arbitrary_precision(p)
    result = mp.ellipk(1 - p)
    return _to_finite_precision(result)


def ellipkinc(phi, m):
    """Incomplete elliptic integral of the first kind."""
    result = mp.ellipf(phi, m)
    return _to_finite_precision(result)


def ellipj(u, m):
    """Jacobian Elliptic functions."""
    sn = mp.ellipfun("sn", u=u, m=m)
    cn = mp.ellipfun("cn", u=u, m=m)
    dn = mp.ellipfun("dn", u=u, m=m)
    phi = mp.asin(sn)
    return _to_finite_precision(sn, cn, dn, phi)


def erf(x):
    """Error function."""
    result = mp.erf(x)
    return _to_finite_precision(result)


def erfc(x):
    """Complementary error function 1 - erf(x)."""
    result = mp.erfc(x)
    return _to_finite_precision(result)


def erfcinv(x):
    """Inverse of the complementary error function."""
    result = mp.erfinv(mp.one - mp.mpf(x))
    return _to_finite_precision(result)


def exp1(x):
    """Exponential integral E1."""
    result = mp.e1(x)
    return _to_finite_precision(result)


def exp10(x):
    """Compute 10**x."""
    result = mp.mpf(10)**x
    return _to_finite_precision(result)


def exp2(x):
    """Compute 2**x."""
    result = mp.mpf(2)**x
    return _to_finite_precision(x)


def expi(x):
    """Exponential integral Ei."""
    result = mp.ei(x)
    return _to_finite_precision(result)


def expit(x):
    """Expit (a.k.a logistic sigmoid)."""
    result = mp.sigmoid(x)
    return _to_finite_precision(x)


def expm1(x):
    """exp(x) - 1."""
    result = mp.expm1(x)
    return _to_finite_precision(result)


def expn(n, x):
    """Generalized exponential integral En."""
    result = mp.expint(n, x)
    return _to_finite_precision(result)


def exprel(x):
    """Relative error exponential, (exp(x) - 1)/x."""
    x = _to_arbitrary_precision(x)
    result = (mp.exp(x) - 1)/x
    return _to_finite_precision(x)


def fdtr(dfn, dfd, x):
    """F cumulative distribution function."""
    dfn, dfd, x  = _to_arbitrary_precision(dfn, dfd, x)
    x_dfn = x * dfn
    result = mp.betainc(
        dfn/2, dfd/2, 0, x_dfn / (dfd + x_dfn), regularized=True
    )
    return _to_finite_precision(result)


def fdtrc(dfn, dfd, x):
    """F survival function."""
    dfn, dfd, x  = _to_arbitrary_precision(dfn, dfd, x)
    x_dfn = x * dfn
    result = mp.betainc(
        dfn/2, dfd/2, x_dfn / (dfd + x_dfn), 1.0, regularized=True
    )
    return _to_finite_precision(result)


def fresnel(x):
    """Fresnel integrals."""
    S, C = mp.fresnels(x), mp.fresnelc(x)
    return _to_finite_precision(S, C)


def gamma(x):
    """Gamma function."""
    if (x == 0.0):
        return math.copysign(math.inf, x)
    if (x < 0 and x == int(x)):
        return math.nan
    result = mp.gamma(x)
    return _to_finite_precision(result)


def gammainc(a, x):
    """Regularized lower incomplete gamma function."""
    result = mp.gammainc(a, 0, x, regularized=True)
    return _to_finite_precision(result)


def gammaincc(a, x):
    """Regularized upper incomplete gamma function."""
    result = mp.gammainc(a, x, mp.inf, regularized=True)
    return _to_finite_precision(result)


def gammaln(x):
    """Logarithm of the absolute value of the gamma function."""
    result = mp.log(abs(mp.gamma(x)))
    return _to_finite_precision(result)


def gammasgn(x):
    """Sign of the gamma function."""
    if (x == 0.0):
        return math.copysign(1.0, x)
    if (x < 0 and x == int(x)):
        return math.nan
    return mp.sign(mp.gamma(x))


def gdtr(a, b, x):
    """Gamma distribution cumulative distribution function."""
    a, b, x = _to_arbitrary_precision(a, b, x)
    result = mp.gammainc(b, 0, a*x, regularized=True)
    return _to_finite_precision(result)


def gdtrc(a, b, x):
    """Gamma distribution survival function."""
    a, b, x = _to_arbitrary_precision(a, b, x)
    result = mp.gammainc(b, a*x, mp.inf, regularized=True)
    return _to_finite_precision(result)


def hankel1(v, z):
    """Hankel function of the first kind."""
    result = mp.hankel1(v, z)
    return _to_finite_precision(result)


def hankel1e(v, z):
    """Exponentially scaled Hankel function of the first kind."""
    v, z = _to_arbitrary_precision(v, z)
    result = mp.hankel1(v, z) * mp.exp(z * -1j)
    return _to_finite_precision(result)


def hankel2(v, z):
    """Hankel function of the second kind."""
    result = mp.hankel2(v, z)
    return _to_finite_precision(result)


def hankel2e(v, z):
    """Exponentially scaled Hankel function of the second kind."""
    v, z = _to_arbitrary_precision(v, z)
    result = mp.hankel2(v, z) * mp.exp(z * 1j)
    return _to_finite_precision(result)


def hyp1f1(a, b, z):
    """Confluent hypergeometric function 1F1."""
    result = mp.hyp1f1(a, b, z)
    return _to_finite_precision(result)


def hyp2f1(a, b, c, z):
    """Gauss hypergeometric function 2F1(a, b; c; z)."""
    result = mp.hyp2f1(a, b, c, z)
    return _to_finite_precision(result)


def hyperu(a, b, z):
    """Confluent hypergeometric function U."""
    result = mp.hyperu(a, b, z)
    return _to_finite_precision(a, b, z)


def kei(x):
    """Kelvin function kei."""
    result = mp.kei(0, x)
    return _to_finite_precision(result)


def ker(x):
    """Kelvin function ker."""
    result = mp.ker(0, x)
    return _to_finite_precision(result)


def lambertw(z, k):
    """Lambert W function."""
    result = mp.lambertw(z, k=k)
    return _to_finite_precision(result)


def lanczos_sum_expg_scaled(z):
    """Exponentially scaled Lanczos approximation to the Gamma function."""
    z = _to_arbitrary_precision(z)
    g = mp.mpf("6.024680040776729583740234375")
    result = (mp.e / (z + g - 0.5))**(z - 0.5) * mp.gamma(z)
    return _to_finite_precision(result)


def lgam1p(x):
    """Logarithm of gamma(x + 1)."""
    x = _to_arbitrary_precision(x)
    result = mp.log(mp.gamma(x + 1))
    return _to_finite_precision(result)


def log1p(x):
    """Logarithm of x + 1."""
    result = mp.log1p(x)
    return _to_finite_precision(x)


def log1pmx(x):
    """log(x + 1) - x."""
    x = _to_arbitrary_precision(x)
    result = mp.log1p(x) - x
    return _to_finite_precision(result)


def loggamma(z):
    """Principal branch of the logarithm of the gamma function."""
    result = mp.loggamma(z)
    return _to_finite_precision(result)


def log_wright_bessel(a, b, x):
    """Natural logarithm of Wright's generalized Bessel function."""
    result = mp.log(_wright_bessel(a, b, x))
    return _to_finite_precision(result)


def nbdtr(k, n, p):
    """Negative binomial cumulative distribution function."""
    k, n, p = _to_arbitrary_precision(k, n, p)
    result = mp.betainc(n, k + 1, 0, p)
    return _to_finite_precision(result)


def nbdtrc(k, n, p):
    """Negative binomial survival function."""
    k, n, p = _to_arbitrary_precision(k, n, p)
    result = mp.betainc(n, k + 1, p, 1)
    return _to_finite_precision(result)


def ndtr(x):
    """Cumulative distribution of the standard normal distribution."""
    result = mp.ncdf(x)
    return _to_finite_precision(x)


def owens_t(h, a):
    """Owen's T Function."""
    h, a = _to_arbitrary_precision(h, a)

    def integrand(x):
        return mp.exp(-h**2 * (1 + x**2) / 2) / (1 + x**2)

    result = mp.quad(integrand, [0, a]) / (2*mp.pi)
    return _to_finite_precision(result)


def pdtr(k, m):
    """Poisson cumulative distribution function."""
    k, m = mp.floor(k), mp.mpf(m)

    def term(j):
        return m**j / mp.factorial(j)

    result =  mp.exp(-m) * mp.nsum(term, [0, k])
    return _to_finite_precision(result)


def pdtrc(k, m):
    """Poisson survival function."""
    k, m = mp.floor(k), mp.mpf(m)
    result = mp.gammainc(k + 1, 0, m, regularized=True)
    return _to_finite_precision(result)


def poch(z, m):
    """Pochhammer symbol."""
    result = mp.rf(z, m)
    return _to_finite_precision(result)


def rgamma(z):
    """Reciprocal of the gamma function."""
    if (x == 0.0):
        return x
    if (x < 0 and x == int(x)):
        return 0.0
    result = mp.one / mp.gamma(z)
    return _to_finite_precision(result)


def riemann_zeta(z):
    """Riemann zeta function."""
    if (z == 1.0):
        return math.nan
    result = mp.zeta(z)
    return _to_finite_precision(result)


def scaled_exp1(x):
    """Exponentially scaled exponential integral E1."""
    x = _to_arbitrary_precision(x)
    result = mp.exp(x)*x*mp.e1(x)
    return _to_finite_precision(result)


def shichi(x):
    """Hyperbolic sine and cosine integrals."""
    shi, chi = mp.shi(x), mp.chi(x)
    return _to_finite_precision(shi), _to_finite_precision(chi)


def sici(x):
    """Sine and cosine integrals."""
    si, ci = mp.si(x), mp.ci(x)
    return _to_finite_precision(si), _to_finite_precision(ci)


def sinpi(x):
    """Sine of pi*x."""
    result = mp.sinpi(x)
    return _to_finite_precision(result)


def spence(z):
    """Spence's function, also known as the dilogarithm."""
    result = mp.polylog(2, 1 - z)
    return _to_finite_precision(result)


def struveh(v, x):
    """Struve function."""
    result = mp.struveh(v, x)
    return _to_finite_precision(v, x)


def struvel(v, x):
    """Modified Struve function."""
    result = mp.struvel(v, x)
    return _to_finite_precision(result)


def tandg(x):
    """Tangent of angle x given in degrees."""
    result = mp.tan(mp.radians(x))
    return _to_finite_precision(result)


def wright_bessel(a, b, x):
    """Wright's generalized Bessel function."""
    result = _wright_bessel(a, b, x)
    return _to_finite_precision(result)


def zeta(z, q):
    """Hurwitz zeta function."""
    if (z == 1.0):
        return math.nan
    result = mp.zeta(z, a=q)
    return _to_finite_precision(result)


def zetac(z):
    """Riemann zeta function minus 1."""
    if (z == 1.0):
        return math.nan
    result = mp.zeta(z) - mp.one
    return _to_finite_precision(result)


def _wright_bessel(a, b, x):
    a, b, x = _to_arbitrary_precision(a, b, x)

    def term(k):
        return x**k / (mp.factorial(k) * mp.gamma(a*k + b))

    return mp.nsum(term, [0, mp.inf])
