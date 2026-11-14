from __future__ import annotations

import asyncio
import math
import os
import time
import typing
from collections import deque
from pathlib import Path

import ezmsg.core as ez
import neo.rawio.baserawio
import numpy as np
import sparse
from ezmsg.baseproc.protocols import processor_state
from ezmsg.baseproc.stateful import BaseStatefulProducer
from ezmsg.baseproc.units import BaseProducerUnit
from ezmsg.util.messages.axisarray import AxisArray, replace


class NeoIteratorSettings(ez.Settings):
    """Settings for :obj:`NeoIterator`."""

    filepath: os.PathLike
    chunk_dur: float = 0.05
    self_terminating: bool = True
    t_offset: typing.Optional[float] = None
    """Time given to the start of the file. ``None`` uses ``time.time()`` when the file is
    opened. Every stream's times are the file's own, shifted by this."""

    gap_tolerance_ms: typing.Optional[float] = 10.0
    """Blackrock files only: the longest gap in the recording's timestamps, in ms, that is
    still treated as continuous. The recording is split at longer gaps (pauses, dropped
    packets) into neo segments, and each one is emitted as a new
    :attr:`~ezmsg.util.messages.axisarray.AxisArray.segment`, on which stateful processors
    downstream reset. ``None`` is neo's own default: a file with any gap raises an error."""


@processor_state
class NeoIteratorState:
    t_offset: float = 0.0
    t_start: float = 0.0
    """Start of the current neo segment, in file time."""
    file_t_start: float = 0.0
    """Start of the first neo segment, in file time: the instant ``t_offset`` is given to."""
    seg_ix: int = 0
    n_segs: int = 0
    first_segment: int = 0
    """The output ``segment`` of neo segment 0."""
    chunk_ix: int = 0
    n_chunks: int = 0
    reader: neo.rawio.baserawio.BaseRawIO | None = None
    streams: dict | None = None
    deque: deque | None = None


class NeoIterator(BaseStatefulProducer[NeoIteratorSettings, AxisArray, NeoIteratorState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Eagerly initialize so metadata is available immediately after construction.
        self._reset_state()
        self._hash = 0

    @property
    def exhausted(self) -> bool:
        return self._state.seg_ix >= self._state.n_segs and not self._state.deque

    def _reset_state(self) -> None:
        self._state.t_offset = self.settings.t_offset if self.settings.t_offset is not None else time.time()
        self._state.t_start = float(np.inf)
        # Reopening the file continues nothing: carry on numbering where the last pass stopped.
        self._state.first_segment = getattr(self, "_next_segment", 0)
        self._state.seg_ix = 0
        self._state.n_segs = 0
        self._state.chunk_ix = 0
        self._state.n_chunks = 0
        self._state.streams = {}
        self._state.deque = deque()
        self._state.reader = None
        self._preload()

    def _preload(self) -> None:
        fpath = Path(self.settings.filepath)
        if not fpath.exists():
            raise FileNotFoundError(f"File not found: {fpath}")

        if fpath.suffix == ".vhdr":
            from neo.rawio import BrainVisionRawIO as RawIO
        elif fpath.suffix.startswith(".ns") or fpath.suffix == ".nev":
            from neo.rawio import BlackrockRawIO as RawIO
        else:
            raise ValueError(f"Unsupported file type: {fpath.suffix}")

        if RawIO.__name__ == "BlackrockRawIO":
            reader = RawIO(filename=str(fpath), gap_tolerance_ms=self.settings.gap_tolerance_ms)
        else:
            reader = RawIO(filename=str(fpath))
        reader.parse_header()

        nb_block = reader.block_count()
        if nb_block > 1:
            raise NotImplementedError("Only single-block files are supported.")

        self._state.reader = reader
        self._state.n_segs = reader.segment_count(0)
        self._next_segment = self._state.first_segment + self._state.n_segs
        streams: dict = self._state.streams

        # analogsignal streams
        nb_sig_streams = reader.signal_streams_count()
        for strm_ix in range(nb_sig_streams):
            nb_chans = reader.signal_channels_count(strm_ix)
            fs = reader.get_signal_sampling_rate(strm_ix)
            # This stream's channels only: the header lists every stream's together.
            chan_struct_arr = reader.header["signal_channels"]
            if "stream_id" in (chan_struct_arr.dtype.names or ()):
                stream_id = reader.header["signal_streams"][strm_ix]["id"]
                chan_struct_arr = chan_struct_arr[chan_struct_arr["stream_id"] == stream_id]
            key = reader.header["signal_streams"][strm_ix]["name"]
            ch_ax = AxisArray.CoordinateAxis(data=chan_struct_arr["name"], dims=["ch"], unit="label")
            # Compute the channel fingerprint once, now. It is cached on the axis
            # and pickled with it, and every message from this stream reuses this
            # same axis object, so one checksum covers the whole file. Left cold it
            # would be computed by the first stateful consumer in this process --
            # and, since unpickling builds a new axis object per message, by the
            # first consumer in every other process, on every message.
            ch_ax.fingerprint
            template = AxisArray(
                data=np.zeros((0, nb_chans), dtype=float),
                dims=["time", "ch"],
                axes={
                    "time": AxisArray.TimeAxis(fs=fs, offset=0.0),
                    "ch": ch_ax,
                },
                key=key,
                # Messages append along `time`; `ch` describes the stream.
                stream_dim="time",
            )
            streams[key] = {
                "idx": strm_ix,
                "type": "analogsignal",
                "t_start": 0.0,  # set per neo segment, in `_start_segment`
                "template": template,
                "prev_samp": 0,
            }

        # event streams
        nb_event_channel = reader.event_channels_count()
        if nb_event_channel > 0:
            # TODO: Event should probably use SampleTriggerMessage
            streams["events"] = {
                "type": "event",
                "nchan": nb_event_channel,
                "template": AxisArray(
                    data=np.array([""]),
                    dims=["time"],
                    axes={"time": AxisArray.CoordinateAxis(data=np.array([0]), dims=["time"], unit="s")},
                    key="events",
                    # Irregular, but still the dimension events accumulate along.
                    # Deliberately not primed: its values are per-message, and the
                    # stream axis is the one axis consumers do not digest.
                    stream_dim="time",
                ),
            }

        # spiketrain streams
        nb_unit = reader.spike_channels_count()
        if nb_unit > 0:
            spk_chans = reader.header["spike_channels"]
            if "wf_sampling_rate" in spk_chans.dtype.names:
                spike_fs = spk_chans["wf_sampling_rate"][0]
            else:
                spike_fs = 30_000.0
            if "name" in spk_chans.dtype.names:
                spk_ch_labels = spk_chans["name"]
            else:
                spk_ch_labels = np.arange(1, 1 + nb_unit).astype(str)

            unit_ax = AxisArray.CoordinateAxis(data=spk_ch_labels, dims=["unit"], unit="unit")
            unit_ax.fingerprint  # primed once for the file -- see the signal stream above
            streams["spike"] = {
                "type": "spiketrain",
                "nchan": nb_unit,
                "template": AxisArray(
                    data=sparse.SparseArray((nb_unit, 0)),
                    dims=["unit", "time"],
                    axes={
                        "unit": unit_ax,
                        "time": AxisArray.TimeAxis(fs=spike_fs, offset=0.0),
                    },
                    key="spike",
                    # Spikes accumulate along `time` even though it trails here;
                    # `unit` describes the sorted units and is what consumers key on.
                    stream_dim="time",
                ),
            }

        self._start_segment(0)
        self._state.file_t_start = self._state.t_start

    def _start_segment(self, seg_ix: int) -> None:
        """Point the chunk counter and each signal stream at neo segment ``seg_ix``."""
        state = self._state
        reader = state.reader
        state.seg_ix = seg_ix
        state.chunk_ix = 0
        state.n_chunks = 0
        if seg_ix >= state.n_segs:
            return
        t_start = np.inf
        t_stop = -np.inf
        for strm in state.streams.values():
            if strm["type"] != "analogsignal":
                continue
            fs = 1 / strm["template"].axes["time"].gain
            strm["t_start"] = reader.get_signal_t_start(0, seg_ix, strm["idx"])
            strm["prev_samp"] = 0
            t_start = min(t_start, strm["t_start"])
            t_stop = max(t_stop, strm["t_start"] + reader.get_signal_size(0, seg_ix, strm["idx"]) / fs)
        if not np.isfinite(t_start):
            # No signal streams (e.g. a lone .nev): the segment's own bounds.
            t_start = reader.segment_t_start(0, seg_ix)
            t_stop = reader.segment_t_stop(0, seg_ix)
        state.t_start = t_start
        state.n_chunks = int(np.ceil((t_stop - t_start) / self.settings.chunk_dur))

    def _chunk_step(self) -> None:
        state = self._state
        reader = state.reader
        seg_ix = state.seg_ix
        segment = state.first_segment + seg_ix
        # File time -> output time: the start of the file lands on `t_offset`.
        t_shift = state.t_offset - state.file_t_start
        t_range = (np.arange(2) + state.chunk_ix) * self.settings.chunk_dur + state.t_start

        for key, strm in state.streams.items():
            if strm["type"] == "analogsignal":
                fs = 1 / strm["template"].axes["time"].gain
                prev_samp = strm["prev_samp"]
                next_samp = max(0, int((t_range[1] - strm["t_start"]) * fs))
                next_samp = min(next_samp, reader.get_signal_size(0, seg_ix, strm["idx"]))
                dat = reader.get_analogsignal_chunk(
                    seg_index=seg_ix,
                    stream_index=strm["idx"],
                    i_start=prev_samp,
                    i_stop=max(next_samp, prev_samp),
                )
                if dat.size:
                    dat = reader.rescale_signal_raw_to_float(dat, dtype=float, stream_index=strm["idx"])
                    msg = replace(
                        strm["template"],
                        data=dat,
                        axes={
                            **strm["template"].axes,
                            "time": replace(
                                strm["template"].axes["time"],
                                offset=t_shift + strm["t_start"] + prev_samp / fs,
                            ),
                        },
                        segment=segment,
                    )
                    state.deque.append(msg)
                strm["prev_samp"] = max(next_samp, prev_samp)

            elif strm["type"] == "event":
                # TODO: Event should probably use SampleTriggerMessage
                for ev_ch_ix in range(strm["nchan"]):
                    ev_timestamps, ev_durations, ev_labels = reader.get_event_timestamps(
                        block_index=0,
                        seg_index=seg_ix,
                        event_channel_index=ev_ch_ix,
                        t_start=t_range[0],
                        t_stop=t_range[1],
                    )
                    if len(ev_timestamps) == 0:
                        continue
                    ev_times = reader.rescale_event_timestamp(ev_timestamps, dtype=float)
                    msg = replace(
                        strm["template"],
                        data=ev_labels,
                        axes={
                            **strm["template"].axes,
                            "time": replace(
                                strm["template"].axes["time"],
                                data=ev_times + t_shift,
                            ),
                        },
                        segment=segment,
                    )
                    state.deque.append(msg)

            elif strm["type"] == "spiketrain":
                samp_step = strm["template"].axes["time"].gain
                # round, not truncate: 0.05 s of 30 kHz is 1499.99... samples in floating point, and a
                # chunk one sample short of its own duration leaves a hole between it and the next.
                n_times = int(round((t_range[1] - t_range[0]) / samp_step))
                tvec = t_range[0] + np.arange(n_times) * samp_step
                samp_idx = np.array([], dtype=int)
                chan_idx = np.array([], dtype=int)
                for spk_ch_ix in range(strm["nchan"]):
                    spike_times = reader.get_spike_timestamps(
                        block_index=0,
                        seg_index=seg_ix,
                        spike_channel_index=spk_ch_ix,
                        t_start=t_range[0],
                        t_stop=t_range[1],
                    )
                    spike_times = reader.rescale_spike_timestamp(spike_times, dtype="float64")
                    samp_idx = np.hstack((samp_idx, np.searchsorted(tvec, spike_times)))
                    chan_idx = np.hstack((chan_idx, np.full((len(spike_times),), spk_ch_ix, dtype=int)))
                    # raw_waveforms = reader.get_spike_raw_waveforms(block_index=0, seg_index=0, spike_channel_index=0,
                    #                                                t_start=0, t_stop=10)
                    # float_waveforms = reader.rescale_waveforms_to_float(
                    #     raw_waveforms, dtype='float32', spike_channel_index=0)
                result = sparse.COO(
                    np.vstack((chan_idx, samp_idx)),
                    data=1,
                    shape=(strm["nchan"], len(tvec)),
                )
                msg = replace(
                    strm["template"],
                    data=result,
                    axes={
                        **strm["template"].axes,
                        "time": replace(strm["template"].axes["time"], offset=t_range[0] + t_shift),
                    },
                    segment=segment,
                )
                state.deque.append(msg)

        state.chunk_ix += 1
        if state.chunk_ix >= state.n_chunks:
            self._start_segment(seg_ix + 1)

    async def _produce(self) -> AxisArray | None:
        state = self._state
        if not state.deque:
            if state.seg_ix >= state.n_segs:
                return None
            self._chunk_step()
        if not state.deque:
            return None
        return state.deque.popleft()

    def __next__(self) -> AxisArray:
        result = self()
        if result is None:
            raise StopIteration
        return result


class NeoIteratorUnit(BaseProducerUnit[NeoIteratorSettings, AxisArray, NeoIterator]):
    SETTINGS = NeoIteratorSettings

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)
    OUTPUT_TERM = ez.OutputStream(typing.Any)

    @ez.publisher(OUTPUT_SIGNAL)
    async def produce(self) -> typing.AsyncGenerator:
        while True:
            out = await self.producer.__acall__()
            if out is not None:
                if math.prod(out.data.shape) > 0:
                    # TODO: Direct msg to OUTPUT_TRIGGER if type is SampleTriggerMessage
                    yield self.OUTPUT_SIGNAL, out
                await asyncio.sleep(0)
            elif self.producer.exhausted:
                break

        ez.logger.debug(f"File ({self.SETTINGS.filepath}) exhausted.")
        if self.SETTINGS.self_terminating:
            raise ez.NormalTermination
        yield self.OUTPUT_TERM, ez.Flag
