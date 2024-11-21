import functools
import math
import numpy as np
import typing

from mpmath import mp  # type: ignore
from typing import overload


__all__ = ["get_signatures", "reference_implementation"]


def signature_from_type_hints(type_hints):
    input_types = []
    for key, val in type_hints.items():
        if key != "return":
            if val in [int, float, complex]:
                input_types.append(val)
                continue
            else:
                raise ValueError
        if val in [float, complex]:
            output_types = (val, )
            continue
        if (
                not isinstance(val, typing._GenericAlias)
                or typing.get_origin(val) is not tuple
        ):
            raise ValueError
        output_types = typing.get_args(val)
    return (tuple(input_types), output_types)


def process_args(func, *args):
    """Convert finite precision arguments to arbitrary precison."""
    expected_signatures = get_signatures(func)
    input_signature = tuple(type(arg) for arg in args)
    if input_signature not in expected_signatures:
        raise ValueError
    output_types = expected_signatures[input_signature]
    new_args = []
    for x, type_ in zip(args, input_signature):
        if type_ is float:
            if x == 0:
                # mpmath doesn't have signed zeros, so if zero, keep this
                # a float to maintain the sign.
                new_args.append(x)
                continue
            new_args.append(mp.mpf(x))
        elif type_ is int:
            new_args.append(mp.mpf(x))
        elif type_ is complex:
            z = mp.mpc(x)
            # Work around for lack of signed zeros in mpmath.
            if (x == 0):
                new_args.append(x)
                continue
            # If real and/or imaginary part is zero, convert to a vanishingly
            # small quantity, preserving the sign of zero.
            if x.real == 0:
                z += mp.mpc(f"1e-{10**500}") * math.copysign(1, x.real)
            if x.imag == 0:
                z += mp.mpc("0.0", f"1e-{10**500}") * math.copysign(1, x.imag)
            new_args.append(z)
    return tuple(new_args), output_types


def process_output(args, output_types):
    """Convert arbitrary precision arguments to finite precision."""
    output = []
    for arg, output_type in zip(args, output_types):
        if isinstance(arg, mp.mpc):
            if output_type is complex:
                output.append(complex(arg))
            else:
                output.append(math.nan)
        else:
            if output_type is float:
                output.append(float(arg))
            else:
                # We expect a complex result but got a real output value. If
                # it's a nan, make sure imaginary part is nan too.
                if mp.isnan(arg):
                    output.append(complex(math.nan, math.nan))
                else:
                    output.append(complex(arg))
    output = tuple(output)
    return output[0] if len(output) == 1 else output


def get_signatures(func):
    type_hints = typing.get_type_hints(func)
    if type_hints:
        input_types, output_types = signature_from_type_hints(
            typing.get_type_hints(func)
        )
        return {input_types: output_types}
    signatures = {}
    for overload in typing.get_overloads(func):
        input_types, output_types = signature_from_type_hints(
            typing.get_type_hints(overload)
        )
        signatures[input_types] = output_types
    return signatures


def reference_implementation(func):
    overloads = typing.get_overloads(func)
    annotations_code = ""
    for overload_def in overloads:
        type_hints = overload_def.__annotations__
        return_type = type_hints['return']
        return_type = (
            return_type.__name__ if return_type in [float, complex]
            else return_type
        )
        input_types = (
            (key, val.__name__) for key, val in type_hints.items()
            if key != 'return'
        )
        annotations_code += (
            "@overload\n"
            "def wrapper("
            f"{','.join((f'{arg}: {type_}' for arg, type_ in input_types))})"
            f" -> {return_type}: ...\n"
        )
    exec(annotations_code)
    @functools.wraps(func)
    def wrapper(*args):
        args, output_types = process_args(func, *args)
        result = func(*args)
        if not isinstance(result, tuple):
            result = (result, )
        return process_output(result, output_types)

    wrapper.__annotations__ =  typing.get_type_hints(func)
    return wrapper


def random_floating_point_numbers(
        min_exp,
        max_exp,
        /,
        size=1,
        *,
        include_negative=True,
        rng=None,
        precision="double"
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

    if include_negative:
        sign = rng.integers(0, 2, size=size, dtype=uint_dtype)
    else:
        sign = np.zeros(size, dtype=uint_dtype)

    sign <<= (num_exponent_bits + num_mantissa_bits)
    biased_exponents <<= num_mantissa_bits

    return (sign | biased_exponents | mantissas).view(dtype=dtype)
