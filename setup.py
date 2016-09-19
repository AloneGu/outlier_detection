"""
"""

import sys
__title__ = 'outlier_detector'
__version__ = '0.1.0'
__author__ = 'Wei Gu'

try:
    from setuptools import setup, find_packages
except ImportError:
    print '%s now needs setuptools in order to build.' % __title__
    print 'Install it using your package manager (usually python-setuptools) or via pip \
            (pip install setuptools).'
    sys.exit(1)

setup(
        name=__title__,
        version=__version__,
        author=__author__,

        install_requires=[
            'luminol',
            'numpy>=1.11.1',
            'pandas>=0.18.1',
            'scikit-learn>=0.16',
            'matplotlib',
            'hdbscan'
        ],
        package_dir={__title__: 'lib/%s' % __title__},
        packages=find_packages('lib'),
        test_suite="test",
        zip_safe=False,
        dependency_links=[
            'git+ssh://git@github.com:linkedin/luminol.git#egg=luminol',

        ]
        )