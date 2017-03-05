import numpy as np
import unittest

from src.tests.util import numpy_array_equals

def test_area_generator(test, fn):
    def check_gen(expected, shape, step_size, asp_ratio, max_asp_ratio_div):
        res = list(fn(shape, step_size, asp_ratio, max_asp_ratio_div))
        test.assertEqual(len(res), len(set(res)))
        test.assertEqual(set(res), set(expected))

    expected = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1),
                (0, 1, 0, 1), (0, 1, 1, 1),
                (1, 0, 1, 0), (1, 0, 1, 1),
                (1, 1, 1, 1)]
    check_gen(expected, (2, 2), 1, 1.0, np.inf)
    
    expected = [(0, 0, 1, 1), (0, 0, 1, 3), 
                (0, 0, 3, 1), (0, 0, 3, 3),
                (0, 2, 1, 3), (0, 2, 3, 3),
                (2, 0, 3, 1), (2, 0, 3, 3),
                (2, 2, 3, 3)]
    check_gen(expected, (4, 4), 2, 1.0, np.inf)

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


def test_compute_integral_image(test, fn):
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([[1.0, 5.0], [10.0, 30.0]])
    test.assertEqual(fn(image, exp=2), expected)

    image = np.ones((5,5))
    expected = np.array([[1,  2,  3,  4,  5],
                         [2,  4,  6,  8, 10],
                         [3,  6,  9, 12, 15],
                         [4,  8, 12, 16, 20],
                         [5, 10, 15, 20, 25]])
    test.assertEqual(fn(image), expected)

    image = np.ones((5,5,5))
    expected = np.repeat(np.expand_dims(expected, axis=2), 5, axis=2)
    test.assertEqual(fn(image), expected)


def test_integral_image_sum(test, fn):
    integral_image = np.array([[[1], [ 2], [ 3], [ 4], [ 5]],
                               [[2], [ 4], [ 6], [ 8], [10]],
                               [[3], [ 6], [ 9], [12], [15]],
                               [[4], [ 8], [12], [16], [20]],
                               [[5], [10], [15], [20], [25]]])
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
        test.assertEqual(fn(integral_image, area), value, msg)

    integral_image = np.array([[[1, 1], [2, 2]],
                               [[2, 2], [4, 4]]])
    tests = [((0, 0, 0, 0), [1, 1]),
             ((0, 0, 1, 1), [4, 4]),
             ((0, 0, 0, 1), [2, 2]),
             ((0, 0, 1, 0), [2, 2])]
    for area, value in tests:
        msg = 'Area {} does not produce expected value {}'.format(area, 
                                                                  value)
        test.assertEqual(fn(integral_image, area), np.array(value), msg)

    integral_image = np.array([[[np.nan], [1.0]],
                               [[np.nan], [np.inf]]])
    tests = [((0, 0, 0, 0), [0.0]),
             ((0, 0, 0, 1), [0.0]),
             ((0, 0, 1, 0), [1.0]),
             ((0, 0, 1, 1), [np.inf])]
    for area, value in tests:
        msg = 'Area {} does not produce expected value {}'.format(area, 
                                                                  value)
        test.assertEqual(fn(integral_image, area), np.array(value), msg)


def test_compute_area_score(test, fn, AML_EXP):
    from src.search.localization import _compute_integral_image
    from sklearn.preprocessing import normalize
    image = np.array([[[0, 1], [1, 0]],
                      [[10, 0], [10, 10]]])
    integral_image = _compute_integral_image(image, AML_EXP)
    tests = [([[20.0, 0.0]], (0, 0, 0, 0), 0.0),
             ([[-1.0, 0.0]], (1, 0, 1, 0), -1.0),
             ([[10.0, 10.0]], (1, 1, 1, 1), 1.0)]
    for query, area, expected in tests:
        msg = 'Query {} with area {} does not produce ' \
              'expected value {}'.format(query, area, expected)
        score = fn(normalize(np.array(query)), area, integral_image)
        test.assertEqual(score, expected, msg)


def test_area_refinement(test, fn, AML_EXP):
    from src.search.localization import _compute_integral_image
    from src.search.localization import _compute_area_score
    from sklearn.preprocessing import normalize
    # Note that in some pathological cases, the best possible region 
    # is not detected even but a region containing this region, 
    # even though the best region should have a bigger score.
    # This is likely due to a combination of integral images and 
    # approximate max pooling, which produces rounding errors.
    # This problem could be solved by checking if the two scores 
    # are equal under some epsilon before checking for the larger score. 
    # If the two scores are equal, the smaller region can then be chosen.
    # This case can be constructed in a unittest, but some tests have 
    # shown that the problem seems to not occur in practice, thus we 
    # save some runtime by omitting the above checks.
    image = np.array([[[0.0, 1.0], [0.1, 0.0], [0.0, 0.5]],
                      [[1.0, 2.0], [1.0, 2.0], [4.0, 4.1]],
                      [[3.0, 0.0], [0.1, 4.0], [5.0, 5.0]]])
    integral_image = _compute_integral_image(image, AML_EXP)
    tests = [([[5., 5.]], (0, 0, 0, 0), (2, 2, 2, 2)),
             ([[0., 1.]], (0, 0, 0, 0), (0, 0, 0, 0)),
             ([[3., 4.]], (0, 0, 0, 0), (0, 2, 1, 2))]

    for query, area, expected in tests:
        msg = 'Query {} with area {} does not produce ' \
              'expected value {}'.format(query, area, expected)
        query = normalize(np.array(query))
        score = _compute_area_score(query, area, integral_image)
        refined_area = fn(query, area, score, integral_image)
        test.assertEqual(refined_area, expected, msg)


def test_localize(test, fn):
    from sklearn.preprocessing import normalize
    # Note that the localize results are dependent on the exact order 
    # the area generator generates the areas
    features = np.array([[0.1, 0.1, 0.1],
                         [0.1, 0.1, 0.1],
                         [0.1, 0.1, 10.]])
    features = np.expand_dims(features, axis=2)
    bbox = fn(normalize(np.array([[10.]])), features, (1, 1), 1, 1.)
    test.assertEqual(bbox, (0, 0, 0, 0))  # Cosine similarity is useless in 
                                          # the one-dimensional case...

    map1 = np.array([[5, 1, 3],
                     [1, 1, 1],
                     [1, 1, 1]])
    map2 = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [3, 1, 10]])
    features = np.dstack((map1, map2))
    bbox = fn(normalize(np.array([[1., 10.]])), features, 
                    (1, 1), 1, 1.)
    test.assertEqual(bbox, (2, 2, 2, 2))


class TestLocalization(unittest.TestCase):
    def setUp(self):
        eq_fn = lambda a, e, msg: numpy_array_equals(self, a, e, msg)
        self.addTypeEqualityFunc(np.ndarray, eq_fn)

    def test_area_generator(self):
        from src.search.localization import _area_generator
        test_area_generator(self, _area_generator)

    def test_compute_integral_image(self):
        from src.search.localization import _compute_integral_image
        test_compute_integral_image(self, _compute_integral_image)

    def test_integral_image_sum(self):
        from src.search.localization import _integral_image_sum
        test_integral_image_sum(self, _integral_image_sum)

    def test_compute_area_score(self):
        from src.search.localization import _compute_area_score, AML_EXP
        test_compute_area_score(self, _compute_area_score, AML_EXP)

    def test_area_refinement(self):
        from src.search.localization import _area_refinement, AML_EXP
        test_area_refinement(self, _area_refinement, AML_EXP)

    def test_localize(self):
        from src.search.localization import localize
        test_localize(self, localize)


class TestLocalizationJIT(unittest.TestCase):
    def setUp(self):
        eq_fn = lambda a, e, msg: numpy_array_equals(self, a, e, msg)
        self.addTypeEqualityFunc(np.ndarray, eq_fn)

    def test_area_generator(self):
        from src.search.localization_jit import _area_generator
        test_area_generator(self, _area_generator)

    def test_compute_integral_image(self):
        from src.search.localization_jit import _compute_integral_image
        test_compute_integral_image(self, _compute_integral_image)

    def test_integral_image_sum(self):
        from src.search.localization_jit import _integral_image_sum
        test_integral_image_sum(self, _integral_image_sum)

    def test_compute_area_score(self):
        from src.search.localization_jit import _compute_area_score, AML_EXP
        test_compute_area_score(self, _compute_area_score, AML_EXP)

    def test_area_refinement(self):
        from src.search.localization_jit import _area_refinement, AML_EXP
        test_area_refinement(self, _area_refinement, AML_EXP)

    def test_localize(self):
        from src.search.localization_jit import localize
        test_localize(self, localize)

if __name__ == '__main__':
    unittest.main()
