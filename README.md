# ezmsg-neo

Load and stream data from [neo](https://github.com/NeuralEnsemble/python-neo) files into [ezmsg](https://github.com/ezmsg-org/ezmsg).

## Installation

`pip install ezmsg-neo`

Or install the latest development version:

`pip install git+https://github.com/ezmsg-org/ezmsg-neo@dev`

### Dependencies

* [`ezmsg`](https://github.com/ezmsg-org/ezmsg)
* [`neo`](https://neo.readthedocs.io/)
* [`sparse`](https://sparse.pydata.org/)

## Usage

ezmsg-neo modules are available under `import ezmsg.neo`

Add the `NeoIteratorUnit` to your ezmsg graph as a data source. You may be interested in other ezmsg extensions for processing the data, such as [`ezmsg-sigproc`](https://github.com/ezmsg-org/ezmsg-sigproc) and [`ezmsg-event`](https://github.com/ezmsg-org/ezmsg-event).

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

### Standalone

The ``NeoIterator`` class can be used outside ezmsg for offline processing:

```python
from ezmsg.neo.source import NeoIterator, NeoIteratorSettings
from ezmsg.util.messages.axisarray import AxisArray

settings = NeoIteratorSettings(filepath="data.nev", chunk_dur=0.05)
neo_iter = NeoIterator(settings)

# Filter messages by key (e.g., "ns3" for analog signals, "spike" for spikes)
sig_msgs = [msg for msg in neo_iter if msg.key.startswith("ns")]

# Concatenate all chunks into a single AxisArray
full_signal = AxisArray.concatenate(*sig_msgs, dim="time")
```

### NeoIterator Messages

The `NeuoIteratorUnit` and `NeoIterator` objects, when called, return `AxisArray` messages with a `key` attribute identifying the signal type:

* **Analog signals** - Key matches the stream name (e.g., `"ns3"`). Data shape: `(n_samples, n_channels)`
* **Spike trains** - Key is `"spike"`. Data is a sparse array with shape `(n_units, n_samples)`
* **Events** - Key is `"events"`. Data contains event labels and its `time` axis is irregular (`CoordinateAxis`) with timestamps associated with each event.

## Development

1. Clone this repo and `cd` into it
2. `uv sync` to setup your environment
3. `uv run pytest tests` to run the tests
4. (Optional) Install pre-commit hooks: `uv run pre-commit install`
