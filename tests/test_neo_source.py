import asyncio
import math
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.baseproc import BaseStatefulTransformer, BaseTransformerUnit, processor_state
from ezmsg.util.messages.axisarray import AxisArray, compute_config_id
from neo.rawio.blackrockrawio import BlackrockRawIO

from ezmsg.neo.source import NeoIterator, NeoIteratorSettings, NeoIteratorUnit


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
    # t_offset=0: times are seconds from the start of the file.
    settings = NeoIteratorSettings(filepath=local_path, t_offset=0.0)
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
        # Within a sample of the file's spike times, counted from the start of the file.
        # (The file's times are epoch-scale, so comparing them directly would pass on anything.)
        assert np.allclose(ez_times, spike_times - reader.segment_t_start(0, 0), rtol=0, atol=2 / 30_000)


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


class TestConfigIdAndEndOfStream:
    @staticmethod
    def _path():
        return Path(__file__).parents[0] / "data" / "blackrock" / "20231027-125608-001.nev"

    def test_every_message_declares_its_config_id(self, monkeypatch):
        import ezmsg.neo.source as source

        calls = []
        monkeypatch.setattr(source, "compute_config_id", lambda m: calls.append(m) or compute_config_id(m))
        it = NeoIterator(NeoIteratorSettings(filepath=self._path()))
        msgs = list(it)
        assert len(calls) == len(it.state.streams)  # once per stream, not per message
        by_key = {}
        for m in msgs:
            assert m.config_id != 0 and m.config_id == compute_config_id(m)
            by_key.setdefault(m.key, set()).add(m.config_id)
        assert all(len(ids) == 1 for ids in by_key.values())
        assert len(set.union(*by_key.values())) == len(by_key) == 2  # signal and spike

    def test_the_file_ends_with_one_marker_per_stream(self):
        msgs = list(NeoIterator(NeoIteratorSettings(filepath=self._path())))
        for key in {m.key for m in msgs}:
            *data, last, eos = [m for m in msgs if m.key == key]
            assert eos.data.shape[eos.get_axis_idx("time")] == 0
            assert all(math.prod(m.data.shape) for m in data + [last])
            assert eos.config_id == last.config_id and eos.segment == last.segment + 1
            n = last.data.shape[last.get_axis_idx("time")]
            assert eos.axes["time"].offset == pytest.approx(last.axes["time"].offset + n * last.axes["time"].gain)

    def test_the_unit_publishes_the_markers_and_a_buffering_stage_flushes(self):
        unit = NeoIteratorUnit(NeoIteratorSettings(filepath=self._path(), self_terminating=False))
        unit.create_producer()
        hold = HoldUnit(HoldSettings())
        hold.create_processor()

        async def run():
            published, released = [], []
            async for stream, m in unit.produce():
                if stream is unit.OUTPUT_SIGNAL:
                    published.append(m)
                    if m.key.startswith("ns"):
                        released += [out async for _, out in hold.on_signal(m)]
            return published, released

        published, released = asyncio.run(run())
        sig = [m for m in published if m.key.startswith("ns")]
        assert sig[-1].data.shape[0] == 0 and sig[-2].data.shape[0] > 0
        assert {m.key for m in published if math.prod(m.data.shape) == 0} == {m.key for m in published}
        # Everything the holding stage took in came out once the marker arrived.
        assert sum(len(m.data) for m in released) == sum(len(m.data) for m in sig)


class HoldSettings(ez.Settings):
    pass


@processor_state
class HoldState:
    held: AxisArray | None = None


class Hold(BaseStatefulTransformer[HoldSettings, AxisArray, AxisArray, HoldState]):
    """Emits each message one message late; ``flush`` releases the one it holds."""

    def _reset_state(self, message: AxisArray) -> None:
        self._state.held = None

    def _process(self, message: AxisArray) -> AxisArray | None:
        out, self._state.held = self._state.held, message
        return out

    def flush(self, message: AxisArray | None = None) -> AxisArray | None:
        out, self._state.held = self._state.held, None
        return out


class HoldUnit(BaseTransformerUnit[HoldSettings, AxisArray, AxisArray, Hold]):
    SETTINGS = HoldSettings
