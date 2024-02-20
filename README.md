[![PyPI version](https://badge.fury.io/py/neighpy.svg)](https://badge.fury.io/py/neighpy) [![test](https://github.com/auggiemarignier/neighpy/actions/workflows/tests.yaml/badge.svg)](https://github.com/auggiemarignier/neighpy/actions/workflows/tests.yaml) [![docs](https://readthedocs.org/projects/neighpy/badge/?version=latest)](https://neighpy.readthedocs.io/en/latest/?badge=latest)

# neighpy

``neighphy`` is a Python implementation of the Neighbourhood Algorithm for the optimisation and appraisal of high-dimensional loss surfaces.
First presented in two papers by M. Sambridge at the Australian National University in 1999, it has since been widely used, particularly in the geophysical community, for the optimisation of complex, high-dimensional functions.

This implementation hopes to replace the original Fortran code with a more modern, user-friendly and flexible Python package.
It is a very simple implementation, with just two classes to implement the two phases of the algorithm: the `neighpy.search.NASearcher` class for the optimisation phase, and the `neighpy.appraise.NAAppraiser` class for the appraisal phase

## Installation

```bash
pip install neighpy
```

## Basic Usage

```python
import numpy as np
from neighpy import NASearcher, NAAppraiser

def objective(x):
    # Objective function to be minimised
    return np.linalg.norm(data - predict_data(x))

# Bounds of the parameter space
bounds = ((-5, 5), (-5, 5))

# Initialise direct search phase
searcher = NASearcher(
    objective,
    ns=100, # number of samples per iteration
    nr=10, # number of cells to resample
    ni=100, # size of initial random search
    n=20, # number of iterations
    bounds=bounds
)

# Run the direct search phase
searcher.run() # results stored in searcher.samples and searcher.objectives

# Initialise the appraisal phase
appraiser = NAAppraiser(
    searcher.samples, # points of parameter space already sampled
    np.exp(-searcher.objectives), # objective function values (as a probability distribution)
    bounds=bounds,
    n_resample=500000, # number of desired new samples
    n_walkers=10 # number of parallel walkers
)

# Run the appraisal phase
appraiser.run()  # Results stored in appraiser.samples
```

## Licence

This code is distributed under a [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contributing

If you have any questions, please to open an issue in this repository.
