from pathlib import Path

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from neo.rawio.blackrockrawio import BlackrockRawIO

from ezmsg.neo.source import NeoIterator, NeoIteratorSettings


def test_neo_iterator_raw():
    local_path = Path(__file__).parents[0] / "data" / "blackrock" / "20231027-125608-001.nev"
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
    local_path = Path(__file__).parents[0] / "data" / "blackrock" / "20231027-125608-001.nev"
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


class TestMessagesArriveReadyForConsumers:
    """Two things only the source can supply, both set once per file.

    ``stream_dim`` names the dimension messages accumulate along -- the one whose
    length is just however much of the file this chunk covered, and which a
    consumer must leave out of the state it caches against the stream's
    configuration. ``fingerprint`` is a coordinate axis's content digest, cached
    on the axis and pickled with it; priming it here spares the first consumer
    in every process from recomputing it on every message.
    """

    @staticmethod
    def _messages():
        local_path = Path(__file__).parents[0] / "data" / "blackrock" / "20231027-125608-001.nev"
        return list(NeoIterator(NeoIteratorSettings(filepath=local_path)))

    def test_every_message_declares_its_stream_dim(self):
        msgs = self._messages()
        assert msgs, "no messages produced"
        undeclared = sorted({m.key for m in msgs if m.stream_dim != "time"})
        assert not undeclared, f"streams not declaring stream_dim='time': {undeclared}"

    def test_the_signal_channel_axis_is_primed(self):
        sig = next(m for m in self._messages() if m.key.startswith("ns"))
        assert "_fingerprint" in sig.axes["ch"].__dict__
        assert sig.axes["ch"].fingerprint is not None

    def test_the_spike_unit_axis_is_primed(self):
        spk = next(m for m in self._messages() if m.key.startswith("spike"))
        assert "_fingerprint" in spk.axes["unit"].__dict__
        assert spk.axes["unit"].fingerprint is not None

    def test_the_chunk_axis_is_left_cold(self):
        """Digesting per-message timestamps would be pure cost: no consumer reads
        the stream axis's fingerprint."""
        spk = next(m for m in self._messages() if m.key.startswith("spike"))
        assert "_fingerprint" not in spk.axes["time"].__dict__

    def test_it_all_survives_the_transport(self):
        import pickle

        sig = next(m for m in self._messages() if m.key.startswith("ns"))
        landed = pickle.loads(pickle.dumps(sig))
        assert landed.stream_dim == "time"
        assert "_fingerprint" in landed.axes["ch"].__dict__
