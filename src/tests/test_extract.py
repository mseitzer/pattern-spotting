import numpy as np
import unittest


def numpy_array_equals(test, actual, expected, msg=None):
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError:
        raise test.failureException(msg)


class TestExtract(unittest.TestCase):
    def setUp(self):
        eq_fn = lambda a, e, msg: numpy_array_equals(self, a, e, msg)
        self.addTypeEqualityFunc(np.ndarray, eq_fn)

    def test_mac(self):
        from src.features.extract import _compute_mac
        features = np.array([
            [[8.30, 8.69, 0.86, 8.65],
             [9.63, 5.32, 5.54, 6.19],
             [1.32, 7.99, 9.61, 1.66]],

            [[4.75, 6.30, 6.93, 7.29],
             [6.91, 6.99, 3.64, 8.78],
             [5.69, 9.13, 7.97, 0.33]],

            [[2.07, 5.49, 8.64, 7.98],
             [2.07, 7.16, 1.80, 3.44],
             [0.19, 9.13, 4.79, 8.52]]])
        expected = np.array([9.63, 9.13, 9.61, 8.78])
        self.assertEqual(_compute_mac(features), expected)


if __name__ == '__main__':
    unittest.main()
