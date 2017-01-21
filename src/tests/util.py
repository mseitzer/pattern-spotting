import numpy as np

def numpy_array_equals(test, actual, expected, msg=None):
    """unittest comparison function for numpy arrays"""
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError:
        raise test.failureException(msg)
