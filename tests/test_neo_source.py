from pathlib import Path

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from neo.rawio.blackrockrawio import BlackrockRawIO

from ezmsg.neo.source import NeoIterator, NeoIteratorSettings


def test_neo_iterator():
    local_path = Path(__file__).parents[0] / "data" / "blackrock" / "20231027-125608-001.nev"
    settings = NeoIteratorSettings(filepath=local_path)
    neo_iter = NeoIterator(settings)

    sig_msgs = [msg for msg in neo_iter if isinstance(msg.axes["time"], AxisArray.LinearAxis)]
    cat = AxisArray.concatenate(*sig_msgs, dim="time")

    reader = BlackrockRawIO(filename=str(local_path))
    reader.parse_header()
    dat = reader.get_analogsignal_chunk(
        seg_index=0,
        stream_index=0,
    )
    dat = reader.rescale_signal_raw_to_float(dat, dtype=float)

    assert np.array_equal(cat.data, dat)
