from pathlib import Path

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from neo.rawio.blackrockrawio import BlackrockRawIO

from ezmsg.neo.source import NeoIterator, NeoIteratorSettings


def test_neo_iterator_raw():
    local_path = (
        Path(__file__).parents[0] / "data" / "blackrock" / "20231027-125608-001.nev"
    )
    settings = NeoIteratorSettings(filepath=local_path)
    neo_iter = NeoIterator(settings)

    sig_msgs = [msg for msg in neo_iter if msg.key.startswith("ns")]
    cat = AxisArray.concatenate(*sig_msgs, dim="time")

    reader = BlackrockRawIO(filename=str(local_path))
    reader.parse_header()
    dat = reader.get_analogsignal_chunk(
        seg_index=0,
        stream_index=0,
    )
    dat = reader.rescale_signal_raw_to_float(dat, dtype=float)

    assert np.array_equal(cat.data, dat)


def test_neo_iterator_spike():
    local_path = (
        Path(__file__).parents[0] / "data" / "blackrock" / "20231027-125608-001.nev"
    )
    settings = NeoIteratorSettings(filepath=local_path)
    neo_iter = NeoIterator(settings)

    spk_msgs = [msg for msg in neo_iter if msg.key.startswith("spike")]
    cat = AxisArray.concatenate(*spk_msgs, dim="time")
    # sparse.concatenate([_.data for _ in spk_msgs], axis=1)

    # Prepare the original
    reader = BlackrockRawIO(filename=str(local_path))
    reader.parse_header()

    # Spot check a few channels
    for ch_ix in [12, 34, 67]:
        spike_times = reader.get_spike_timestamps(
            block_index=0,
            seg_index=0,
            spike_channel_index=ch_ix,
        )
        spike_times = reader.rescale_spike_timestamp(spike_times, dtype="float64")

        # inds = cat.data[ch_ix].nonzero()[0]
        inds = cat.data[ch_ix].coords[0]
        ez_times = cat.axes["time"].value(inds)
        assert np.allclose(ez_times, spike_times)
