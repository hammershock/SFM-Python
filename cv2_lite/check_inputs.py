import inspect
from functools import wraps

import numpy as np


def check_input_shapes(func):
    """
    A decorator to check input shapes and consistency based on the function's signature.
    The parameter names are used to determine the expected shapes.
    """
    # Get the signature of the function
    sig = inspect.signature(func)
    param_shapes = {
        'point3ds': (None, 3),  # Any number of rows, 3 columns
        'point2ds': (None, 2),  # Any number of rows, 2 columns
        'K': (3, 3)  # Camera matrix must be 3x3
    }

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind the function arguments to the signature
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        arguments = bound_args.arguments
        for name, value in arguments.items():
            # Check if the parameter name has a predefined shape
            if name in param_shapes:
                expected_shape = param_shapes[name]
                # Check if the argument is a numpy array
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"Argument {name} must be a numpy ndarray, but got {type(value)}.")
                # Check the shape of the numpy array
                if len(value.shape) != len(expected_shape):
                    raise ValueError(f"Argument {name} must be a {len(expected_shape)}-dimensional array.")
                for dim, (actual, expected) in enumerate(zip(value.shape, expected_shape)):
                    if expected is not None and actual != expected:
                        raise ValueError(f"Dimension {dim} of argument {name} must be {expected}, but got {actual}.")

        # Additional check for the length of point3ds and point2ds to be equal
        if 'point3ds' in arguments and 'point2ds' in arguments:
            if arguments['point3ds'].shape[0] != arguments['point2ds'].shape[0]:
                raise ValueError("The number of 3D points (point3ds) must match the number of 2D points (point2ds).")

        return func(*args, **kwargs)

    return wrapper
