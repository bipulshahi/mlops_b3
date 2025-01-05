import io
import os
from pathlib import Path

from setuptools import setup, find_packages

#Metadata of package
NAME = 'Myloanapp'
DESCRIPTION = 'Loan Prediction Model'
URL = 'https://github.com/bipulshahi'
EMAIL = 'bipul@abc.com'
AUTHOR = 'Bipul Kumar'
REQUIRES_PYTHON = '>=3.7.0'

pwd = os.path.abspath(os.path.dirname(__file__))

# Get the list of packages to be installed
def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(pwd, fname), encoding='utf-8') as f:
        return f.read().splitlines()
    

try:
    with io.open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the Package __version__.py module
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME

about = {}
with io.open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    include_package_data=True,                      # Corrected spelling from 'include_pacakge_data'
    package_data={
        '': ['*.txt', '*.md', '*.py', 'datasets/*.csv', 'trained_models/*.pkl', 'processing/*.py', 'VERSION'],
    },
    install_requires=list_reqs(),
    extras_require={},  # Corrected spelling from 'extra_requires'
    license='ABC',  # Corrected spelling from 'licence'
    classifiers=[
        'License :: OSI Approved :: ABC License',  # Corrected spelling from 'LICENCE'
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
