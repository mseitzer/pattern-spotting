import numpy as np
import unittest

from src.tests.util import numpy_array_equals

class TestLocalization(unittest.TestCase):
    def setUp(self):
        eq_fn = lambda a, e, msg: numpy_array_equals(self, a, e, msg)
        self.addTypeEqualityFunc(np.ndarray, eq_fn)

    def test_area_generator(self):
        def check_gen(expected, shape, step_size, asp_ratio, max_asp_ratio_div):
            res = list(_area_generator(shape, step_size, 
                                       asp_ratio, max_asp_ratio_div))
            self.assertEqual(len(res), len(set(res)))
            self.assertEqual(set(res), set(expected))

        from src.search.localization import _area_generator
        expected = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1),
                    (0, 1, 0, 1), (0, 1, 1, 1),
                    (1, 0, 1, 0), (1, 0, 1, 1),
                    (1, 1, 1, 1)]
        check_gen(expected, (2, 2), 1, None, 1.0)
        
        expected = [(0, 0, 1, 1), (0, 0, 1, 3), 
                    (0, 0, 3, 1), (0, 0, 3, 3),
                    (0, 2, 1, 3), (0, 2, 3, 3),
                    (2, 0, 3, 1), (2, 0, 3, 3),
                    (2, 2, 3, 3)]
        check_gen(expected, (4, 4), 2, None, 1.0)

        expected = [(0, 0, 1, 1), (0, 0, 3, 3),
                    (0, 2, 1, 3), (2, 0, 3, 1),
                    (2, 2, 3, 3)]
        check_gen(expected, (5, 5), 2, 1.0, 1.0)

        # Allow aspect ratios 1:2 to 1:1 to 2:1
        expected = [(0, 0, 1, 1), (0, 0, 1, 3), (0, 0, 3, 1), (0, 0, 3, 3),
                    (0, 2, 1, 3), (0, 2, 3, 3),
                    (2, 0, 3, 1), (2, 0, 3, 3),
                    (2, 2, 3, 3)]
        check_gen(expected, (5, 5), 2, 1.0, 2.0)


    def test_compute_integral_image(self):
        from src.search.localization import _compute_integral_image
        image = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.array([[1.0, 5.0], [10.0, 30.0]])
        self.assertEqual(_compute_integral_image(image, exp=2), expected)

        image = np.ones((5,5))
        expected = np.array([[1,  2,  3,  4,  5],
                             [2,  4,  6,  8, 10],
                             [3,  6,  9, 12, 15],
                             [4,  8, 12, 16, 20],
                             [5, 10, 15, 20, 25]])
        self.assertEqual(_compute_integral_image(image), expected)

        image = np.ones((5,5,5))
        expected = np.repeat(np.expand_dims(expected, axis=2), 5, axis=2)
        self.assertEqual(_compute_integral_image(image), expected)

    def test_integral_image_sum(self):
        from src.search.localization import _integral_image_sum
        integral_image = np.array([[1,  2,  3,  4,  5],
                                   [2,  4,  6,  8, 10],
                                   [3,  6,  9, 12, 15],
                                   [4,  8, 12, 16, 20],
                                   [5, 10, 15, 20, 25]])
        tests = [((0, 0, 0, 0), 1),
                 ((0, 0, 0, 4), 5),
                 ((0, 0, 4, 0), 5),
                 ((0, 0, 4, 4), 25),
                 ((1, 1, 3, 3), 9),
                 ((2, 2, 4, 4), 9),
                 ((0, 1, 2, 4), 12),
                 ((2, 1, 3, 3), 6),
                 ((0, 3, 4, 4), 10),
                 ((0, 0, 1, 4), 10)]
        for area, value in tests:
            msg = 'Area {} does not produce expected value {}'.format(area, 
                                                                      value)
            self.assertEqual(_integral_image_sum(integral_image, area),
                             value, msg)

        integral_image = np.array([[[1, 1], [2, 2]],
                                   [[2, 2], [4, 4]]])
        tests = [((0, 0, 0, 0), [1, 1]),
                 ((0, 0, 1, 1), [4, 4]),
                 ((0, 0, 0, 1), [2, 2]),
                 ((0, 0, 1, 0), [2, 2])]
        for area, value in tests:
            msg = 'Area {} does not produce expected value {}'.format(area, 
                                                                      value)
            self.assertEqual(_integral_image_sum(integral_image, area),
                             np.array(value), msg)

        integral_image = np.array([[np.nan, 1.0],
                                   [np.nan, np.inf]])
        tests = [((0, 0, 0, 0), [0.0]),
                 ((0, 0, 0, 1), [0.0]),
                 ((0, 0, 1, 0), [1.0])]
        for area, value in tests:
            msg = 'Area {} does not produce expected value {}'.format(area, 
                                                                      value)
            self.assertEqual(_integral_image_sum(integral_image, area),
                             np.array(value), msg)
        self.assertNotEqual(_integral_image_sum(integral_image, (0, 0, 1, 1)),
                            np.inf)

    def test_compute_area_score(self):
        from src.search.localization import _compute_area_score
        from src.search.localization import _compute_integral_image, AML_EXP
        from sklearn.preprocessing import normalize
        integral_image = np.array([[[0, 1], [1, 0]],
                                   [[10, 0], [10, 10]]])
        integral_image = _compute_integral_image(integral_image, AML_EXP)
        tests = [([[20.0, 0.0]], (0, 0, 0, 0), 0.0),
                 ([[-1.0, 0.0]], (1, 0, 1, 0), -1.0),
                 ([[10.0, 10.0]], (1, 1, 1, 1), 1.0)]
        for query, area, expected in tests:
            msg = 'Query {} with area {} does not produce ' \
                  'expected value {}'.format(query, area, expected)
            score = _compute_area_score(normalize(np.array(query)), area, 
                                        integral_image)
            self.assertEqual(score, expected, msg)

    def test_localize(self):
        from src.search.localization import localize
        from sklearn.preprocessing import normalize
        # Note that the localize results are dependent on the exact order 
        # the area generator generates the areas
        features = np.array([[0.1, 0.1, 0.1],
                             [0.1, 0.1, 0.1],
                             [0.1, 0.1, 10.]])
        features = np.expand_dims(features, axis=2)
        bbox = localize(normalize(np.array([[10.]])), features, (1, 1), 1, 1.)
        self.assertEqual(bbox, (0, 0, 0, 0))  # Cosine similarity is useless in 
                                              # the one-dimensional case...

        map1 = np.array([[ 5, 1, 3],
                         [ 1, 1, 1],
                         [ 1, 1, 1]])
        map2 = np.array([[ 1, 1, 1],
                         [ 1, 1, 1],
                         [ 3, 1, 10]])
        features = np.dstack((map1, map2))
        bbox = localize(normalize(np.array([[1., 10.]])), features, 
                        (1, 1), 1, 1.)
        self.assertEqual(bbox, (2, 2, 2, 2))

if __name__ == '__main__':
    unittest.main()
