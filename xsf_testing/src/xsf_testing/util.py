import csv
import functools
import inspect
import math
import numpy as np
import os
import polars as pl
import re
import typing

from collections import defaultdict
from mpmath import mp  # type: ignore
from threading import Lock
from typing import overload


__all__ = ["get_signatures", "reference_implementation", "TracedUfunc"]


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
            # Again mpmath doesn't have signed zeros, so if the real or
            # imaginary part are zero, we just pass through a complex
            # float. This should be kept in mind for the cases where
            # this might matter.
            if (x.real == 0 or x.imag) == 0:
                new_args.append(x)
            else:
                new_args.append(mp.mpc(x))
    return tuple(new_args), output_types


def process_output(args, output_types):
    """Convert arbitrary precision arguments to finite precision."""
    output = []
    for arg, output_type in zip(args, output_types):
        if isinstance(arg, mp.mpc):
            if output_type is complex:
                output.append(complex(arg))
            else:
                if abs(arg.imag) == 0:
                    output.append(float(arg))
                else:
                    # Expected a real result, but got complex. Convention is to return
                    # nan in this case.
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


def reference_implementation(*, dps=100, uses_mp=True):
    if not uses_mp:
        # The reference implementation does not use arbitrary precision.
        return lambda func: func

    def _reference_implementation(func):
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
            with mp.workdps(dps):
                args, output_types = process_args(func, *args)
                result = func(*args)
                if not isinstance(result, tuple):
                    result = (result, )
                return process_output(result, output_types)
        wrapper.__annotations__ =  typing.get_type_hints(func)
        setattr(wrapper, "_mp", func)
        return wrapper
    return _reference_implementation


class TracedUfunc:
    def __init__(self, ufunc, /, *, outpath=None):
        self.__ufunc = ufunc
        self.__outpath = outpath
        self.__lock = Lock()
        self._dtype_map = {
            "float32": "f",
            "float64": "d",
            "float128": "g",
            "complex64": "F",
            "complex128": "D",
            "complex256": "G",
            "int64": "p",
            "int32": "i",
        }

    def __call__(self, *args, **kwargs):
        try:
            expanded_args = np.broadcast_arrays(*args)
            # There is a test that inputs will not broadcast which
            # is asserted to raise a ValueError. Just skip that case.
            # It's not relevant when directly testing scalar kernels.
        except ValueError:
            return self.__ufunc(*args, **kwargs)
        dtypes = tuple(val.dtype for val in expanded_args)
        dtypes = self.__ufunc.resolve_dtypes(dtypes + (None, ) * self.__ufunc.nout)
        dtypes = [self._dtype_map[str(dtype)] for dtype in dtypes]
        sig = "".join(dtypes[:self.__ufunc.nin]) + "->"
        sig += "".join(dtypes[self.__ufunc.nin:])
        expanded_args = [val.flatten() for val in expanded_args]
        rows = (
            row + (sig, ) + self._get_file_metadata() for row in zip(*expanded_args)
        )
        with self.__lock:
            with open(self.__outpath, 'a', newline='') as csvfile:
                csv.writer(csvfile, dialect="unix").writerows(rows)

        return self.__ufunc(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.__ufunc, name)

    def _get_file_metadata(self):
        frame = inspect.currentframe()
        pattern1 = re.compile(r"[^/]+/tests/.+\.py")
        pattern2 = re.compile("^test_.*")
        for _ in range(10):
            test_name = frame.f_code.co_name
            test_file = frame.f_globals.get("__file__")
            if test_file is None:
                return test_file, test_name
            test_file = os.path.join(*test_file.split(os.path.sep)[-3:])
            if pattern1.match(test_file) and pattern2.match(test_name):
                return test_file, test_name
            frame = frame.f_back
        return None, None

    @property
    def _outpath(self):
        return self.__outpath


def _parse_column(col, dtype):
    if dtype is np.complex64 or dtype is np.complex128:
        col = col.to_numpy().astype(dtype)
        real = pl.Series(col.real)
        imag = pl.Series(col.imag)
        return pl.DataFrame({"real": real, "imag": imag}).to_struct()
    if dtype is np.float64:
        return col.cast(pl.Float64)
    if dtype is np.float32:
        return col.cast(pl.Float32)
    if dtype is np.int32:
        return col.cast(pl.Int32)
    if dtype is np.int64:
        return col.cast(pl.Int64)
    raise ValueError(f"unsupported dtype: {dtype}")


def _traced_cases_to_parquet(infile, outpath):
    os.makedirs(outpath)
    dtype_map = {
        "f": np.float32,
        "d": np.float64,
        "F": np.complex64,
        "D": np.complex128,
        "p": np.intp,
        "i": np.int32,
    }
    with open(infile, 'r', newline='') as csvfile:
        new_rows = defaultdict(list)
        for row in csv.reader(csvfile, dialect="unix"):
            args = row[:-3]
            types = row[-3]
            in_types, out_types = types.split("->")
            if len(args) != len(in_types):
                continue
            try:
                args = [
                    str(dtype_map[typecode](arg)) for typecode, arg in zip(in_types, args)
                ]
            except IndexError:
                continue
            new_rows[types].append(args)
        
        for types, rows in new_rows.items():
            in_types, _ = types.split("->")
            dtypes = [dtype_map[typecode] for typecode in in_types]
            df = pl.DataFrame(rows, orient="row").unique()
            df.columns = [f"in{i}" for i in range(len(df.columns))]
            df = df.with_columns(
                *(
                    _parse_column(df[colname], dtype).alias(f"in{i}")
                    for i, (colname, dtype) in enumerate(zip(df.columns, dtypes))
                )
            )
            df.write_parquet(
                os.path.join(outpath, f"{types.replace('->', '_')}.parquet"),
                compression="zstd",
                compression_level=22,
            )
