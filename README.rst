BIDIRECTIONAL CROWD FLOWS WITH INTRODUCTION OF OBSTACLES
========================================================

Code for the project of the course Complex System Simulation, UvA 2021.

The contributors of this project are:
- Caterina Buranelli
- Marija Pujic
- Robert van Koesveld
- Iris Reitsma



.. image:: https://travis-ci.org/svenkreiss/socialforce.svg?branch=master
    :target: https://travis-ci.org/svenkreiss/socialforce


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
