Get Started
===========

Installation
------------

.. code-block:: bash

    $ pip install neighpy


Basic Usage
-----------

Each phase of the Neighbourhood Algorithm is implemented as a separate class in the ``neighpy`` package.
Each class has a ``run`` method that performs the algorithm and stores the results in the object.

Direct Search Phase
^^^^^^^^^^^^^^^^^^^

The direct search phase uses ``neighpy.NASearcher`` to first perform a random search of the parameter space, then optimise iteratively around the best found parameters.
``neighpy.NASearcher`` needs an ``objective`` function, which is the function to be optimised.
The algorithm will optimise the function in the sense of `minimising` it.
Also needed are the ``bounds`` of the parameter space, and some tuning parameters.

Tuning parameters:
 * ``ns`` - Number of samples to be generated at each iteration
 * ``nr`` - Resample the parameter space around the best ``nr`` samples
 * ``ni`` - Number of samples from the initial random search
 * ``n`` - Number of iterations to perform

.. code-block:: python

    from neighpy import NASearcher

    def objective(x):
        return x[0]**2 + x[1]**2

    bounds = [(-5, 5), (-5, 5)]
    ns = 10
    nr = 5
    ni = 10
    n = 10

    searcher = NASearcher(objective, ns, nr, ni, n, bounds)
    searcher.run()

The ``run`` method will save all the models and their respective objective values in the ``searcher.samples`` and ``searcher.objectives`` attributes.  You will end up with $n_{i} + n \times n_{s}$ samples and their respective objective values.

.. code-block:: python

    best = searcher.samples[np.argmin(searcher.objectives)]

Full details of the algorithm can be found in `Sambridge, 1999 (I) <https://academic.oup.com/gji/article/138/2/479/596234>`_.


Appraisal Phase
^^^^^^^^^^^^^^^

The appraisal phase is performed by ``neighpy.NAAppraiser``.
It takes the ``searcher.samples`` and ``searcher.objectives`` as input, and uses them to resample the objective function without needing to reevaluate the objective function.

Tuning parameters:
 * ``n_resample`` - Number of new samples to obtain
 * ``n_walkers`` - How many parallel walkers to send to explore the parameter space

.. code-block:: python

    from neighpy import NAAppraiser

    n_resample = 100
    n_walkers = 10

    appraiser = NAAppraiser(searcher.samples, searcher.objectives, bounds, n_resample, n_walkers)
    appraiser.run()

The ``run`` method will save the new samples in the ``appraiser.samples`` attribute, from which one can plot or summarise the objective surface.  
For example, the following will plot a 2D projection of the samples for two parameters.
Denser regions of the plot are regions represent the more optimum regions of the parameter space.

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, ax = fig.add_subplots(1,1)
    ax.scatter(appraiser.samples[:, 0], appraiser.samples[:, 1])
    plt.show()

``run`` also calculates the mean and covariance of the samples on the fly, and stores them in the ``appraiser.mean`` and ``appraiser.covariance`` attributes.
This is useful if you expect the objective function to be Gaussian/unimodal symmetric, and don't want the memory overhead of storing all the samples.

Full details of the algorithm can be found in `Sambridge, 1999 (II) <https://academic.oup.com/gji/article/138/3/727/578730>`_.
