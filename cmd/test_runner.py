#!/usr/bin/env python3
"""Script to execute all of the projects unit tests"""

import os
import subprocess
import sys

proj_dir = os.path.abspath(os.path.split(os.path.realpath(__file__))[0]
                           + '/..')

# Defines all unit test scripts
tests = [
    'src/features/test_extract.py'
]

def main(args):
    results = []
    for test in tests:
        cwd = os.path.join(proj_dir, os.path.dirname(test))
        file = os.path.basename(test)

        res = subprocess.call(["python", file], cwd=cwd)
        results.append(res)

    print('Test summary: ')
    num_tests_passed = 0
    for test, res in zip(tests, results):
        if res == 0:
            res_str = 'PASS'
            num_tests_passed += 1
        else:
            res_str = 'FAIL'
        print('{}: {}'.format(test, res_str))
    print('Passed {}/{} tests'.format(num_tests_passed, len(tests)))

if __name__ == '__main__':
    main(sys.argv[1:])
