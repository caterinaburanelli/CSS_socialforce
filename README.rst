BIDIRECTIONAL CROWD FLOWS WITH INTRODUCTION OF OBSTACLES
========================================================

Code for the project of the course Complex System Simulation, UvA 2021.

The model is based on the Social Force Model for pedestrian dynamics created by Dirk Helbing and Péter Molnár.

The contributors of this project are:

- Caterina Buranelli

- Marija Pujic

- Robert van Koesveld

- Iris Reitsma



.. image:: https://travis-ci.org/svenkreiss/socialforce.svg?branch=master
    :target: https://travis-ci.org/svenkreiss/socialforce


source: Social Force Model
==========================

.. code-block::

    Social force model for pedestrian dynamics
    Dirk Helbing and Péter Molnár
    Phys. Rev. E 51, 4282 – Published 1 May 1995


Install and Run
===============

.. code-block:: sh

    # install from PyPI
    pip install 'socialforce[test,plot]'

    # or install from source
    pip install -e '.[test,plot]'

    # run linting and tests
    pylint socialforce
    pytest tests/*.py


EXPANSIONS:
===========

- inclusion of different obstacles

- removal of periodic movement of agents ( an agent passes the coridor, new agent is randomly placed in the opposite edge )

- different tendency of respawning on the agent's right side of the corridor

- higher repulsive force between the agents and the obstacles

    HOW TO IMPLEMENT:
        - socialforce/potentials.py - line 86 - change to : return self.u0 * np.exp(-0.4 * np.linalg.norm(r_aB, axis=-1) / self.r)

SIMULATIONS:
============

Different setups together with the code for running of the simulations can be found in the 'test' file

    - layouts: 
        'benchmark'
            .. image:: figures/walkway_benchmark_0_new4.gif 
        'horizontal'
            .. image:: figures/walkway_horizontal_0_new4.gif 
        'pillars'
            .. image:: figures/walkway_pillars_0_new4.gif 
        'single'
            .. image:: figures/walkway_single_0_new4.gif 
        'angled'
            .. image:: figures/walkway_angled_0_new4.gif 
        
        - number of people in the system at all times
        
            - number of simulations conducted for statistical accuracy
            
RESULTS:
========

results of the simulations can be plotted by running the read_and_plot.py file

    - the file path has to be edited depednig on which resutls are being accessed
