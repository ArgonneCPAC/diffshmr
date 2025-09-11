diffshmr
============

Installation
------------
To install diffshmr into your environment from the source code::

    $ cd /path/to/root/diffshmr
    $ pip install .

Testing
-------
To run the suite of unit tests::

    $ cd /path/to/root/diffshmr
    $ pytest

To build html of test coverage::

    $ pytest -v --cov --cov-report html
    $ open htmlcov/index.html

