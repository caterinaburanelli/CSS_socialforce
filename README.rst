Social Force Model
==================

.. code-block::

    Social force model for pedestrian dynamics
    Dirk Helbing and Péter Molnár
    Phys. Rev. E 51, 4282 – Published 1 May 1995


Install and Run
===============

.. code-block:: sh

    # install from pypi
    pip install 'socialforce[test,plot]'

    # or install from source
    pip install -e '.[test,plot]'

    # run linting
    pylint socialforce
    pytest


Ped-Space Scenarios
===================

+----------------------------------------+----------------------------------------+
| .. image:: docs/gate.gif               | .. image:: docs/gate.png               |
+----------------------------------------+----------------------------------------+


Ped-Ped Scenarios
=================

+----------------------------------------+----------------------------------------+
| .. image:: docs/crossing.png           | .. image:: docs/narrow_crossing.png    |
+----------------------------------------+----------------------------------------+
| .. image:: docs/opposing.png           | .. image:: docs/2opposing.png          |
+----------------------------------------+----------------------------------------+