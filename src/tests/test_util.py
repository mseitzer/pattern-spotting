import numpy as np
import unittest

from src.tests.util import numpy_array_equals

class TestUtil(unittest.TestCase):
    def setUp(self):
        eq_fn = lambda a, e, msg: numpy_array_equals(self, a, e, msg)
        self.addTypeEqualityFunc(np.ndarray, eq_fn)

    def test_normalize(self):
        from src.util import normalize
        v = np.array([[1.0], [2.0], [3.0]])
        v_ln = np.sqrt(1.0**2 + 2.0**2 + 3.0**2)
        expected = np.array([[1.0 / v_ln], [2.0 / v_ln], [3.0 / v_ln]])
        self.assertEqual(normalize(v), expected)

        v = np.array([[0.0], [0.0], [0.0]])
        expected = np.array([[0.0], [0.0], [0.0]])
        self.assertEqual(normalize(v), expected)


if __name__ == '__main__':
    unittest.main()
