ezmsg.neo
=========

Load and stream data from `neo <https://github.com/NeuralEnsemble/python-neo>`_ files into `ezmsg <https://www.ezmsg.org>`_.

Overview
--------

``ezmsg-neo`` provides components for reading neural data from files supported by the `Python Neo <https://neo.readthedocs.io/>`_ library and streaming them into ezmsg pipelines.

Key features:

* **NeoIterator** - A Python iterator for reading data chunk-by-chunk from Neo-supported file formats
* **NeoIteratorUnit** - An ezmsg Unit that streams file data as :obj:`AxisArray` messages
* **Multiple data types** - Streams analog signals, events, and spike trains from the same file
* **Sparse spike representation** - Spike trains are represented using sparse arrays for memory efficiency

Supported File Formats
^^^^^^^^^^^^^^^^^^^^^^

Currently supported formats:

* **BrainVision** (``.vhdr``)
* **Blackrock** (``.ns*``, ``.nev``)

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install ezmsg-neo

Or install the latest development version:

.. code-block:: bash

   pip install git+https://github.com/ezmsg-org/ezmsg-neo@main

Dependencies
^^^^^^^^^^^^

Core dependencies:

* ``ezmsg`` - Core messaging framework
* ``neo`` - Neural Ensemble IO library for reading neural data files
* ``sparse`` - Sparse array library for efficient spike train representation

Quick Start
-----------

For general ezmsg tutorials and guides, visit `ezmsg.org <https://www.ezmsg.org>`_.

Here's a basic example of using ``NeoIteratorUnit`` to stream data from a file:

.. code-block:: python

   import ezmsg.core as ez
   from ezmsg.neo.source import NeoIteratorUnit
   from ezmsg.util.messages.key import FilterOnKey
   from ezmsg.util.debuglog import DebugLog
   from ezmsg.event.rate import EventRate

   comps = {
       "NEO": NeoIteratorUnit(filepath="path/to/file.nev", chunk_dur=0.05),
       "FILTER": FilterOnKey(key="spike"),
       "RATE": EventRate(bin_duration=0.05),
       "LOG": DebugLog()
   }
   conns = (
       (comps["NEO"].OUTPUT_SIGNAL, comps["FILTER"].INPUT_SIGNAL),
       (comps["FILTER"].OUTPUT_SIGNAL, comps["RATE"].INPUT_SIGNAL),
       (comps["RATE"].OUTPUT_SIGNAL, comps["LOG"].INPUT),
   )
   ez.run(components=comps, connections=conns)

Using NeoIterator Standalone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``NeoIterator`` class can be used outside of ezmsg for offline processing:

.. code-block:: python

   from ezmsg.neo.source import NeoIterator, NeoIteratorSettings
   from ezmsg.util.messages.axisarray import AxisArray

   settings = NeoIteratorSettings(filepath="data.nev", chunk_dur=0.05)
   neo_iter = NeoIterator(settings)

   # Filter messages by key (e.g., "ns3" for analog signals, "spike" for spikes)
   sig_msgs = [msg for msg in neo_iter if msg.key.startswith("ns")]

   # Concatenate all chunks into a single AxisArray
   full_signal = AxisArray.concatenate(*sig_msgs, dim="time")

Output Message Types
^^^^^^^^^^^^^^^^^^^^

``NeoIteratorUnit`` outputs :obj:`AxisArray` messages with a ``key`` attribute identifying the data type:

* **Analog signals** - Key matches the stream name (e.g., ``"ns3"``). Data shape: ``(n_samples, n_channels)``
* **Spike trains** - Key is ``"spike"``. Data is a sparse array with shape ``(n_units, n_samples)``
* **Events** - Key is ``"events"``. Data contains event labels with timestamps in the ``time`` axis

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
