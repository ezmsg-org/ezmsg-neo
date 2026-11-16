"""A file neo splits into segments is emitted as that many ``AxisArray.segment`` values."""

import numpy as np
import pytest

from ezmsg.neo.source import NeoIterator, NeoIteratorSettings

FS = 1000.0
# (start time in s, samples) of each neo segment: 2 s of data, a 3 s pause, 1.5 s of data.
SEGMENTS = [(10.0, 2000), (15.0, 1500)]


class FakeBlackrockRawIO:
    """The slice of the neo RawIO API the iterator uses, for a two-segment recording.

    Sample ``i`` of the whole recording holds the value ``i``.
    """

    last_kwargs: dict = {}

    def __init__(self, filename, **kwargs):
        type(self).last_kwargs = kwargs
        self.header = {
            "signal_channels": np.array([("a",), ("b",)], dtype=[("name", "U8")]),
            "signal_streams": np.array([("ns2",)], dtype=[("name", "U8")]),
        }

    def parse_header(self):
        pass

    def block_count(self):
        return 1

    def segment_count(self, block_index):
        return len(SEGMENTS)

    def signal_streams_count(self):
        return 1

    def signal_channels_count(self, stream_index):
        return 2

    def get_signal_sampling_rate(self, stream_index):
        return FS

    def get_signal_t_start(self, block_index, seg_index, stream_index):
        return SEGMENTS[seg_index][0]

    def get_signal_size(self, block_index, seg_index, stream_index):
        return SEGMENTS[seg_index][1]

    def get_analogsignal_chunk(self, seg_index, stream_index, i_start, i_stop):
        first = sum(n for _, n in SEGMENTS[:seg_index])
        return np.repeat(np.arange(first + i_start, first + i_stop)[:, None], 2, axis=1)

    def rescale_signal_raw_to_float(self, dat, dtype=float, stream_index=None):
        return dat.astype(dtype)

    def event_channels_count(self):
        return 0

    def spike_channels_count(self):
        return 0


FakeBlackrockRawIO.__name__ = "BlackrockRawIO"


@pytest.fixture
def two_segment_file(tmp_path, monkeypatch):
    import neo.rawio

    monkeypatch.setattr(neo.rawio, "BlackrockRawIO", FakeBlackrockRawIO)
    path = tmp_path / "rec.ns2"
    path.touch()
    return path


def test_each_neo_segment_is_a_new_segment(two_segment_file):
    msgs = list(NeoIterator(NeoIteratorSettings(filepath=two_segment_file, chunk_dur=0.5, t_offset=100.0)))
    n_total = sum(n for _, n in SEGMENTS)
    # Every sample once, in order, and none read past a segment's end.
    np.testing.assert_array_equal(np.concatenate([m.data[:, 0] for m in msgs]), np.arange(n_total))
    # Each neo segment's chunks share a segment, one more at the gap with no marker between them,
    # and one empty end-of-stream marker ends the file.
    assert [len(m.data) for m in msgs] == [500] * 4 + [500] * 3 + [0]
    assert [m.segment for m in msgs] == [0] * 4 + [1] * 3 + [2]
    assert_segment_rule(msgs)
    msgs = [m for m in msgs if len(m.data)]
    # The start of the file lands on t_offset; the pause survives in the offsets.
    assert msgs[0].axes["time"].offset == pytest.approx(100.0)
    assert msgs[3].axes["time"].offset == pytest.approx(101.5)
    assert msgs[4].axes["time"].offset == pytest.approx(105.0)


def test_gap_tolerance_is_passed_to_the_blackrock_reader(two_segment_file):
    NeoIterator(NeoIteratorSettings(filepath=two_segment_file))
    assert FakeBlackrockRawIO.last_kwargs == {"gap_tolerance_ms": 10.0}
    NeoIterator(NeoIteratorSettings(filepath=two_segment_file, gap_tolerance_ms=None))
    assert FakeBlackrockRawIO.last_kwargs == {"gap_tolerance_ms": None}


def assert_segment_rule(msgs):
    """Per stream: segments start at 0 and never decrease, a marker is its data's segment + 1, and no
    number is shared by a marker and data or by two runs of data."""
    for key in {m.key for m in msgs}:
        stream = [(m.segment, m.data.shape[m.get_axis_idx("time")]) for m in msgs if m.key == key]
        segs = [seg for seg, _ in stream]
        assert segs == sorted(segs)
        assert next(seg for seg, n in stream if n) == 0
        data_segs = {seg for seg, n in stream if n}
        runs = [seg for i, (seg, n) in enumerate(stream) if n and (i == 0 or stream[i - 1][0] != seg)]
        assert len(runs) == len(set(runs))
        for (prev, n_prev), (seg, n) in zip(stream, stream[1:]):
            if not n and seg != prev:  # an end-of-stream marker
                assert n_prev and seg == prev + 1 and seg not in data_segs


def test_reopening_the_file_continues_numbering_upward(two_segment_file):
    it = NeoIterator(NeoIteratorSettings(filepath=two_segment_file, chunk_dur=0.5))
    first = list(it)
    it._reset_state()
    second = list(it)
    assert_segment_rule(first)
    assert [m.segment for m in second] == [m.segment + 3 for m in first]


def test_two_readers_number_a_file_the_same(two_segment_file):
    settings = NeoIteratorSettings(filepath=two_segment_file, chunk_dur=0.5)
    assert [m.segment for m in NeoIterator(settings)] == [m.segment for m in NeoIterator(settings)]


def test_the_file_ends_with_a_marker(two_segment_file):
    msgs = list(NeoIterator(NeoIteratorSettings(filepath=two_segment_file, chunk_dur=0.5, t_offset=0.0)))
    assert [i for i, m in enumerate(msgs) if not len(m.data)] == [7]
    for last, eos in [(msgs[6], msgs[7])]:
        assert eos.data.shape == (0, 2) and eos.key == last.key
        assert eos.config_id == last.config_id != 0
        assert eos.axes["time"].offset == pytest.approx(last.axes["time"].offset + len(last.data) / FS)


class FakeWithEvents(FakeBlackrockRawIO):
    """One event channel, with an event at each whole second of the recording, labelled with varying widths."""

    def event_channels_count(self):
        return 1

    def get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        t0, n = SEGMENTS[seg_index]
        times = np.arange(np.ceil(max(t_start, t0)), min(t_stop, t0 + n / FS))
        return times, None, np.array(["x" * int(t) for t in times])

    def rescale_event_timestamp(self, ts, dtype=float):
        return ts.astype(dtype)


FakeWithEvents.__name__ = "BlackrockRawIO"


def test_event_messages_declare_one_config_id_and_end_the_file(two_segment_file, monkeypatch):
    import neo.rawio
    from ezmsg.util.messages.axisarray import compute_config_id

    monkeypatch.setattr(neo.rawio, "BlackrockRawIO", FakeWithEvents)
    msgs = list(NeoIterator(NeoIteratorSettings(filepath=two_segment_file, chunk_dur=0.5)))
    events = [m for m in msgs if m.key == "events"]
    assert all(m.config_id == compute_config_id(m) for m in events)
    assert len({m.config_id for m in events}) == 1
    assert {m.config_id for m in events}.isdisjoint({m.config_id for m in msgs if m.key != "events"})
    # 2 events in the first neo segment, 2 in the second (at 15 s and 16 s), then the end marker.
    assert [len(m.data) for m in events] == [1, 1, 1, 1, 0]
    assert [m.segment for m in events] == [0, 0, 1, 1, 2]
    assert_segment_rule(msgs)
