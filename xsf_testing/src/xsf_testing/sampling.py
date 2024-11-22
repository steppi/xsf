import numpy as np


def _random_floating_point_numbers(
        min_exp,
        max_exp,
        /,
        shape=1,
        *,
        precision="double",
        rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    match precision:
        case "double":
            num_exponent_bits = 11
            num_mantissa_bits = 52
            bias = 1023
            dtype = np.float64
            uint_dtype = np.uint64
        case "float":
            num_exponent_bits = 8
            num_mantissa_bits = 23
            bias = 127
            dtype = np.float32
            uint_dtype = np.uint32
        case _:
            raise ValueError(
                "precison must be one of \"double\" or \"float\", "
                f"received {precision}"
            )
    assert min_exp <= max_exp
    assert min_exp >= -bias

    exponents = rng.integers(min_exp, max_exp + 1, size=size)
    biased_exponents = (exponents + bias).astype(uint_dtype)
    mantissas = rng.integers(
        0, 1 << num_mantissa_bits, size=size, dtype=uint_dtype
    )

    biased_exponents <<= num_mantissa_bits
    return (biased_exponents | mantissas).view(dtype=dtype)


def decompose_float(x):
    if isinstance(x, float):
        x = np.float64(x)

    finfo = np.finfo(type(x))
    num_exponent_bits = finfo.nexp
    num_mantissa_bits = finfo.nmant
    bias = 2**(num_exponent_bits - 1) - 1
    dtype = type(x)
    uint_dtype = np.dtype(f"uint{finfo.bits}")

    bits = x.view(uint_dtype)

    sign = (bits >> (num_exponent_bits + num_mantissa_bits)) & 1
    exponent_mask = ((1 << num_exponent_bits) - 1) << num_mantissa_bits
    exponent = (bits & exponent_mask) >> num_mantissa_bits
    mantissa_mask = (1 << num_mantissa_bits) - 1
    mantissa = bits & mantissa_mask
    return int(sign), int(exponent), int(mantissa)


def count_floats_in_range(x, y):
    if isinstance(x, float):
        x = np.float64(x)
    if isinstance(y, float):
        y = np.float64(y)
    assert type(x) == type(y)

    if x == y:
        return 1

    if x < 0 <= y:
        return (
            count_floats_in_range(type(x)(0.0), -x)
            + count_floats_in_range(type(y)(0.0), y)
        )

    if x < y <  0:
        return count_floats_in_range(-y, -x)

    num_mantissa_bits = np.finfo(type(x)).nmant

    sign_x, exp_x, mantissa_x = decompose_float(x)
    sign_y, exp_y, mantissa_y = decompose_float(y)

    largest_mantissa = (1 << num_mantissa_bits) - 1

    if exp_x == exp_y:
        return mantissa_y - mantissa_x + 1


    count_left_bucket = largest_mantissa - mantissa_x + 1
    count_right_bucket = mantissa_y + 1
    count_intermediate = (exp_y - exp_x - 1) * largest_mantissa
    return count_left_bucket + count_intermediate + count_right_bucket


def _random_floating_point_numbers_nonnegative(x, y, /, *, shape=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(x, float):
        x = np.float64(x)
    if isinstance(y, float):
        y = np.float64(y)

    assert x < y

    sign_x, exp_x, mantissa_x = decompose_float(x)
    sign_y, exp_y, mantissa_y = decompose_float(y)

    if sign_x or sign_y:
        raise ValueError("x and y must both be nonnegative")

    finfo = np.finfo(type(x))
    num_exponent_bits = finfo.nexp
    num_mantissa_bits = finfo.nmant
    bias = 2**(num_exponent_bits - 1) - 1
    dtype = type(x)
    uint_dtype = np.dtype(f"uint{finfo.bits}")

    largest_mantissa = (1 << num_mantissa_bits) - 1
    weights = np.ones(exp_y - exp_x + 1, dtype=dtype)
    weights[0] = (mantissa_x + 1) / (largest_mantissa + 1)
    weights[-1] = (mantissa_y + 1) / (largest_mantissa + 1)
    weights /= np.sum(weights)

    exponents = rng.choice(np.arange(exp_x, exp_y + 1, dtype=uint_dtype), p=weights, size=shape)
    min_mantissas = np.zeros(shape, dtype=uint_dtype)
    max_mantissas = np.full(shape, 1 << num_mantissa_bits)
    min_mantissas[exponents == exp_x] = mantissa_x
    max_mantissas[exponents == exp_y] = mantissa_y
    mantissas = rng.integers(min_mantissas, max_mantissas + 1, dtype=uint_dtype)

    return ((exponents << num_mantissa_bits) | mantissas).view(dtype=dtype)


def random_floating_point_numbers(x, y, /, *, shape=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(x, float):
        x = np.float64(x)
    if isinstance(y, float):
        y = np.float64(y)
    assert x < y

    sign_x, exp_x, mantissa_x = decompose_float(x)
    sign_y, exp_y, mantissa_y = decompose_float(y)

    if not (sign_x or sign_y):
        return _random_floating_point_numbers_nonnegative(
            x, y, shape=shape, rng=rng
        )

    if sign_x and sign_y:
        return -_random_floating_point_numbers_nonnegative(
            -y, -x, shape=shape, rng=rng
        )

    num_floats_left = count_floats_in_range(x, type(x)(0.0))
    num_floats_right = count_floats_in_range(type(y)(0.0), y)
    p = num_floats_left / (num_floats_left + num_floats_right)
    sides = rng.binomial(1, p, size=shape)
    result = np.empty(shape)
    num_left = np.sum(sides == 0)
    num_right = len(sides) - num_left
    result[sides == 0] = -_random_floating_point_numbers_nonnegative(
        type(x)(0.0), -x, shape=num_left
    )
    result[sides == 1] = _random_floating_point_numbers_nonnegative(
        type(y)(0.0), y, shape=num_right
    )
    return result


def random_complex_numbers(x0, x1, y0, y1, /, *, shape=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = random_floating_point_numbers(x0, x1, shape=shape, rng=rng)
    y = random_floating_point_numbers(y0, y1, shape=shape, rng=rng)
    z = x + y*1j
    return z


def test_values_from_interval(open_bracket, x, y, close_bracket, /, *, n=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    assert open_bracket in ["[", "("]
    assert close_bracket in ["]", ")"]
    assert x < y
    assert n >= 2
    assert type(x) == type(y)

    eps = np.finfo(type(x)).eps

    endpoints = []
    if open_bracket == "[":
        endpoints.append(x)
    x = np.nextafter(x, np.inf)
    if close_bracket == "]":
        endpoints.append(y)
    y = np.nextafter(y, -np.inf)
    endpoints = np.asarray(endpoints)

    num_remaining_cases = n - len(endpoints)

    intervals = [
        (max(eps, x), y),
        (x, min(-eps, y)),
        (max(x, -eps), min(y, eps)),
    ]
    includes = np.fromiter((a < b for a, b in intervals), dtype=np.bool)

    if np.all(includes):
        proportions = np.array([0.45, 0.45, 0.1])
    elif np.all(includes == np.array([True, True, False])):
        proportions = [0.5, 0.5, 0.0]
    elif np.sum(includes == 2):
        proportions = np.array([0.0, 0.0, 0.1])
        for i, b in enumerate(includes[:-1]):
            if i:
                proportions[i] = 0.9
    else:
        proportions = np.array(includes, dtype=type(x))

    which = rng.choice(np.arange(0, 3), p=proportions, size=num_remaining_cases)
    result = np.empty(num_remaining_cases, dtype=type(x))

    if includes[0]:
        result[which == 0] = random_floating_point_numbers(
            *intervals[0], shape=np.sum(which == 0), rng=rng
        )
    if includes[1]:
        result[which == 1] = random_floating_point_numbers(
            *intervals[1], shape=np.sum(which == 1), rng=rng
        )
    if includes[2]:
        result[which == 2] = random_floating_point_numbers(
            *intervals[2], shape=np.sum(which == 2), rng=rng
        )

    return np.concatenate([endpoints, result])
        
        
    

    








 



