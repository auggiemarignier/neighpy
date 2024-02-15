.. neighpy documentation master file, created by
   sphinx-quickstart on Wed Feb 14 11:52:13 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

neighpy
===================================

``neighphy`` is a Python implementation of the Neighbourhood Algorithm for the optimisation and appraisal of high-dimensional loss surfaces.
First presented in two papers by M. Sambridge at the Australian National University in 1999, it has since been widely used, particularly in the geophysical community, for the optimisation of complex, high-dimensional functions.

This implementation hopes to replace the original Fortran code with a more modern, user-friendly and flexible Python package.
It is a very simple implementation, with just two classes to implement the two phases of the algorithm: the :py:class:`~neighpy.search.NASearcher` class for the optimisation phase, and the :py:class:`~neighpy.appraise.NAAppraiser` class for the appraisal phase.

.. toctree::
    :maxdepth: 1
    :caption: Contents

    getstarted
    api
    examples
    contributing
    license


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
