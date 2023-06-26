#
# MarkovModels setuptools script
#
import os

from setuptools import find_packages, setup


# Load text for description
with open('README.md') as f:
    readme = f.read()

# Load version number
#with open(os.path.join('MarkovModels', 'version.txt'), 'r') as f:
 #   version = f.read()

version = "0"

# Go!
setup(
    # Module name (lowercase)
    name='MarkovModels',

    version=version,
    description='markov models for cardiac modelling',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Joseph Shuttleworth, Dominic Whittaker, Michael Clerx, Maurice Hendrix, Gary Mirams',
    author_email='joseph.shuttleworth@nottingham.ac.uk',
    maintainer='Joseph Shuttleworth',
    maintainer_email='joseph.shuttleworth@nottingham.ac.uk',
    url='https://github.com/joeyshuttleworth/MarkovModels',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],

    # Packages to include
    packages=find_packages(
        include=('MarkovModels', 'MarkovModels.*')),

    # Include non-python files (via MANIFEST.in)
    include_package_data=True,

    # Required Python version
    python_requires='>=3.6',

    # List of dependencies
    install_requires=[
        'pints==0.4.0',
        'scipy==1.9.1',
        'numpy==1.23.5',
        'matplotlib==3.6.2',
        'pandas==1.5.0',
        'sympy==1.11.1',
        'numba==0.56.2',
        'regex==2022.9.13',
        'myokit==1.33.0',
        'seaborn==0.12.0',
        'openpyxl==3.1.2',
        'markov_builder @ git+https://git@github.com/CardiacModelling/Markov-builder@8193d0f10795119e032ee9e458e15f60318594f4',
        'scikit-build==0.16.7',
        'numbalsoda @ git+https://git@github.com/NicholasWogan/numbalsoda@c9183b5bc75002bac5270339f0e0ca645f102b4d'
    ],
    extras_require={
        'test': [
            'pytest-cov>=2.10',     # For coverage checking
            'pytest>=4.6',          # For unit tests
            'flake8>=3',            # For code style checking
            'isort',
            'mock>=3.0.5',         # For mocking command line args etc.
            'codecov>=2.1.3',
        ],
    },
)
