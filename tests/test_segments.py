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
    assert [m.segment for m in msgs] == [0] * 4 + [1] * 3
    # The start of the file lands on t_offset; the pause survives in the offsets.
    assert msgs[0].axes["time"].offset == pytest.approx(100.0)
    assert msgs[3].axes["time"].offset == pytest.approx(101.5)
    assert msgs[4].axes["time"].offset == pytest.approx(105.0)


def test_gap_tolerance_is_passed_to_the_blackrock_reader(two_segment_file):
    NeoIterator(NeoIteratorSettings(filepath=two_segment_file))
    assert FakeBlackrockRawIO.last_kwargs == {"gap_tolerance_ms": 10.0}
    NeoIterator(NeoIteratorSettings(filepath=two_segment_file, gap_tolerance_ms=None))
    assert FakeBlackrockRawIO.last_kwargs == {"gap_tolerance_ms": None}


def test_reopening_the_file_does_not_repeat_a_segment(two_segment_file):
    it = NeoIterator(NeoIteratorSettings(filepath=two_segment_file, chunk_dur=0.5))
    last = list(it)[-1].segment
    it._reset_state()
    assert next(it).segment == last + 1
