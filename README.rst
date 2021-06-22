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

source:
Social Force Model
==================

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


Ped-Ped-Space Scenarios
=======================

+----------------------------------------+----------------------------------------+
| .. image:: docs/separator.gif          | .. image:: docs/gate.gif               |
+----------------------------------------+----------------------------------------+
| Emergent lane formation with           | Emergent lane formation with           |
| 30 pedestrians:                        | 60 pedestrians:                        |
|                                        |                                        |
| .. image:: docs/walkway_30.gif         | .. image:: docs/walkway_60.gif         |
+----------------------------------------+----------------------------------------


EXPANSIONS:
===========

- inclusion of different obstacles

- removal of periodic movement of agents ( an agent passes the coridor, new agent is randomly placed in the opposite edge )

- different tendency of respawning on the agent's right side of the corridor

- wider field of view

- higher repulsive force between the agents and the obstacles

SIMULATIONS:
============

Different setups together with the code for running of the simulations can be found in the 'test' file

    - layouts: 
        'benchmark'
            insert photo
        'horizontal'
            insert photo
        'pillars'
            insert photo
        'single'
            insert photo
        'angled'
            insert photo
        
        - number of people in the system at all times
        
            - number of simulations conducted for statistical accuracy
