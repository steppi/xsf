import atexit
import csv
import functools
import inspect
import math
import numpy as np
import os
import typing

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
        self.__trace = []
        self.__outpath = outpath
        self.__lock = Lock()
        atexit.register(self.dump)

    def __call__(self, *args, **kwargs):
        trace_data = {"args": args, "kwargs": kwargs}
        trace_data.update(self._get_file_metadata())
        self.__trace.append(trace_data)
        return self.__ufunc(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.__ufunc, name)

    @property
    def trace(self):
        output = []
        for entry in self.__trace:
            try:
                args = np.broadcast_arrays(*entry["args"])
            except ValueError:
                # There is a test that inputs will not broadcast which
                # is asserted to raise a ValueError. Just skip that case.
                # It's not relevant when directly testing scalar kernels.
                continue
            args = [val.flatten() for val in args]
            test_file, test_name = entry["test_file"], entry["test_name"]
            output.extend([row + (test_file, test_name) for row in zip(*args)])
        return output

    def _get_file_metadata(self):
        frame = inspect.currentframe().f_back.f_back
        test_file = os.path.basename(frame.f_globals.get("__file__", "???????"))
        test_name = frame.f_code.co_name
        return {"test_file": test_file, "test_name": test_name}

    @property
    def _outpath(self):
        return self.__outpath

    def dump(self):
        if self.__outpath is not None:
            with self.__lock:
                with open(self.__outpath, 'a', newline='') as csvfile:
                    csv.writer(csvfile, lineterminator='\n').writerows(self.trace)
