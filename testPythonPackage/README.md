# PyPi => The python Package index
* The primary repository for third-party python packages. It offers a vast collection of open-source tools and libraries.
* Discovery - Browse and serach PyPi to find packages that meet our needs, from web frameworks to scientific computing libraries.
* Centralised - PyPI is the go-to source for the majority of packages used in the Python ecosystem.
* pip - standard tool for installing, removing and managing python packages.
* key functions
 * Install
 * Uninstall
 * List - Show listed packages
 * Dependency Management - Automatically resolves and install package dependencies

# Why Package?
    * Structure your code in a way that makes it easy to dostribute, reuse and contribute to the Python community.

# setup.py
    * The cornerstone file for Python package configuration, conatining metadata and installation instructions (uses 'setuptools' library)

# Distribution
    * Learn how to upload your packages to Pypi so others can benifit from your work.


hello_world
    |--hello_world
        |--__init__.py
        |--main.py
    |--setup.py

# Virtual environment
* virtualenv ml_package
* ml_package\Scripts\activate
* pip install setuptools twine
# run it at location where setup.py is created
* python setup.py sdist         