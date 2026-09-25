.. _configuration:

Configuration
*************

ArgusSim reads its settings from a TOML file. On import, ``argus_sim`` loads the first ``*.toml`` file in the
current working directory that parses as a configuration and makes it available as ``argus_sim.c``. With no such file
it uses the defaults. ``asim defaults`` writes the defaults to ``argussim.toml`` as a starting point. The file has one
section per data class described below.

Changes to ``argus_sim.c`` in user code apply to objects built after the change. They are not written back to the
file, and objects that already exist keep the values they were built with. For example, a
:class:`argus_sim.CradlePointing` reads ``structure.n_telescopes`` when it is created, so one built before the change
below keeps 1200 telescopes and one built after it has 2:

.. code:: python

    import argus_sim

    argus_sim.c.structure.n_telescopes = 2

.. automodule:: argus_sim.config
