# ezmsg-neo

Load and stream data from [neo](https://github.com/NeuralEnsemble/python-neo) files into [ezmsg](https://github.com/ezmsg-org/ezmsg).

## Installation

`pip install ezmsg-neo`

### Dependencies

* [`ezmsg`](https://github.com/ezmsg-org/ezmsg)
* [`neo`](https://neo.readthedocs.io/)
* [`sparse`](https://sparse.pydata.org/)

## Usage

Add the `NeoIteratorUnit` to your ezmsg graph as a data source. You may be interested in other ezmsg extensions for processing and visualizing the data, such as [`ezmsg-sigproc`](https://github.com/ezmsg-org/ezmsg-sigproc) and [`ezmsg-event`](https://github.com/ezmsg-org/ezmsg-event).

```python
import ezmsg.core as ez
from ezmsg.neo.source import NeoIteratorUnit
from ezmsg.util.messages.key import FilterOnKey
from ezmsg.util.debuglog import DebugLog
from ezmsg.event.rate import EventRate


comps = {
    "NEO": NeoIteratorUnit(filepath="path/to/file", chunk_dur=0.05),
    "FILTER": FilterOnKey(key="spike"),
    "RATE": EventRate(bin_duration=0.05),
    "LOG": DebugLog()  # Print the output to the console
}
conns = (
    (comps["NEO"].OUTPUT_SIGNAL, comps["FILTER"].INPUT_SIGNAL),
    (comps["FILTER"].OUTPUT_SIGNAL, comps["RATE"].INPUT_SIGNAL),
    (comps["RATE"].OUTPUT_SIGNAL, comps["LOG"].INPUT),
)
ez.run(components=comps, connections=conns)

```

## Setup (Development)

1. Clone this repo and `cd` into it
2. `uv sync --all-extras --dev --python 3.10` to setup your environment
3. `uv run pytest tests` to run the tests

ezmsg-neo modules are available under `import ezmsg.neo`
