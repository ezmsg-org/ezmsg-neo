import asyncio
from collections import deque
from dataclasses import replace
import os
from pathlib import Path
import time
import typing

import ezmsg.core as ez

from .util import AxisArray  # This one has labels field.
from ezmsg.util.generator import GenState
import neo.rawio.baserawio
import numpy as np


class NeoIteratorSettings(ez.Settings):
    """Settings for :obj:`NeoIterator`."""

    filepath: os.PathLike
    chunk_dur: float = 0.1
    self_terminating: bool = True
    t_offset: typing.Optional[float] = None


class NeoIterator:
    def __init__(self, settings: NeoIteratorSettings):
        self._settings = settings
        self._reader: typing.Optional[neo.rawio.baserawio.BaseRawIO] = None
        self._playback_state: typing.Optional[dict] = None
        self._reset()

    def _reset(self):
        self._playback_state = {
            "t_offset": self._settings.t_offset
            if self._settings.t_offset is not None
            else time.time(),
            "t_start": np.inf,
            "chunk_ix": 0,
            "msg_queue": deque(),
            "streams": {},
        }
        self._preload()

    def _preload(self):
        fpath = Path(self._settings.filepath)

        if not fpath.exists():
            raise FileNotFoundError(f"File not found: {fpath}")

        if fpath.suffix == ".vhdr":
            from neo.rawio import BrainVisionRawIO as RawIO
        else:
            raise ValueError(f"Unsupported file type: {fpath.suffix}")

        self._reader = RawIO(filename=str(fpath))
        self._reader.parse_header()

        nb_block = self._reader.block_count()
        if nb_block > 1:
            raise NotImplementedError("Only single-block files are supported.")
        nb_seg = [self._reader.segment_count(_) for _ in range(nb_block)][0]
        if nb_seg > 1:
            raise NotImplementedError("Only single-segment files are supported.")
        nb_sig_streams = self._reader.signal_streams_count()

        t_stop = -np.inf

        # Fill out metadata for analogsignal streams
        for strm_ix in range(nb_sig_streams):
            t_start = self._reader.get_signal_t_start(0, 0, strm_ix)
            self._playback_state["t_start"] = min(
                self._playback_state["t_start"], t_start
            )
            nb_chans = self._reader.signal_channels_count(strm_ix)
            fs = self._reader.get_signal_sampling_rate(strm_ix)
            nb_samps = self._reader.get_signal_size(0, 0, strm_ix)
            t_stop = max(t_stop, t_start + nb_samps / fs)
            chan_struct_arr = self._reader.header["signal_channels"]
            key = self._reader.header["signal_streams"][strm_ix]["name"]
            template = AxisArray(
                data=np.zeros((0, nb_chans), dtype=float),
                dims=["time", "ch"],
                axes={
                    "time": AxisArray.Axis.TimeAxis(fs=fs, offset=0.0),
                    "ch": AxisArray.Axis(labels=chan_struct_arr["name"].tolist()),
                },
                key=key,
            )
            self._playback_state["streams"][key] = {
                "idx": strm_ix,
                "type": "analogsignal",
                "t_start": t_start,
                "template": template,
                "prev_samp": 0,
            }

        # Fill out metadata for event streams
        nb_event_channel = self._reader.event_channels_count()
        if nb_event_channel > 0:
            # TODO: Event should probably use SampleTriggerMessage
            self._playback_state["streams"]["events"] = {
                "type": "event",
                "nchan": nb_event_channel,
                "template": AxisArray(
                    data=np.array([""]),
                    dims=["time"],
                    axes={"time": AxisArray.Axis.TimeAxis(fs=1, offset=0.0)},
                    key="events",
                ),
            }

        # TODO: Add support for spiketrain streams
        # nb_unit = self._reader.spike_channels_count()

        t_elapsed = t_stop - self._playback_state["t_start"]
        self._playback_state["n_chunks"] = int(
            np.ceil(t_elapsed / self._settings.chunk_dur)
        )

    def __iter__(self):
        self._reset()
        return self

    def _chunk_step(self):
        state = self._playback_state
        t_range = (
            state["chunk_ix"] * self._settings.chunk_dur,
            (state["chunk_ix"] + 1) * self._settings.chunk_dur,
        )

        for key, strm in state["streams"].items():
            if strm["type"] == "analogsignal":
                # Fetch data from last_idx to next_idx = next_time * fs
                fs = 1 / strm["template"].axes["time"].gain
                prev_samp = strm["prev_samp"]
                next_samp = max(0, int((t_range[1] - strm["t_start"]) * fs))
                dat = self._reader.get_analogsignal_chunk(
                    seg_index=0,
                    stream_index=strm["idx"],
                    i_start=prev_samp,
                    i_stop=next_samp,
                )
                if dat.size:
                    dat = self._reader.rescale_signal_raw_to_float(dat, dtype=float)
                    msg = replace(
                        strm["template"],
                        data=dat,
                        axes={
                            "ch": strm["template"].axes["ch"],
                            "time": replace(
                                strm["template"].axes["time"],
                                offset=t_range[0] + state["t_offset"],
                            ),
                        },
                    )
                    state["msg_queue"].append(msg)
                strm["prev_samp"] = next_samp
            elif strm["type"] == "event":
                # TODO: Event should probably use SampleTriggerMessage
                for ev_ch_ix in range(strm["nchan"]):
                    ev_timestamps, ev_durations, ev_labels = (
                        self._reader.get_event_timestamps(
                            block_index=0,
                            seg_index=0,
                            event_channel_index=ev_ch_ix,
                            t_start=t_range[0],
                            t_stop=t_range[1],
                        )
                    )
                    ev_times = self._reader.rescale_event_timestamp(
                        ev_timestamps, dtype=float
                    )
                    for ev_t, ev_l in zip(ev_times, ev_labels):
                        msg = replace(
                            strm["template"],
                            data=np.array(
                                [ev_l],
                            ),
                            axes={
                                "time": replace(
                                    strm["template"].axes["time"],
                                    offset=ev_t + state["t_offset"],
                                ),
                            },
                        )
                        state["msg_queue"].append(msg)

            else:
                # TODO: spiketrain
                # spike_timestamps = self._reader.get_spike_timestamps(block_index=0, seg_index=0, spike_channel_index=0,
                #                                                t_start=0, t_stop=10)
                # spike_timestamps
                # spike_times = self._reader.rescale_spike_timestamp(spike_timestamps, dtype='float64')
                # raw_waveforms = reader.get_spike_raw_waveforms(block_index=0, seg_index=0, spike_channel_index=0,
                #                                                t_start=0, t_stop=10)
                # float_waveforms = reader.rescale_waveforms_to_float(raw_waveforms, dtype='float32', spike_channel_index=0)
                raise NotImplementedError(f"Unsupported stream type: {strm['type']}")

        state["chunk_ix"] += 1

    def __next__(self) -> AxisArray:
        state = self._playback_state
        if not state["msg_queue"]:
            if state["chunk_ix"] >= state["n_chunks"]:
                # TODO Close file
                raise StopIteration
            self._chunk_step()

        if not state["msg_queue"]:
            raise StopIteration

        return state["msg_queue"].popleft()


class NeoIteratorUnit(ez.Unit):
    STATE = GenState
    SETTINGS = NeoIteratorSettings

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)
    OUTPUT_TERM = ez.OutputStream(typing.Any)

    def initialize(self) -> None:
        self.construct_generator()

    def construct_generator(self):
        self.STATE.gen = NeoIterator(
            settings=self.SETTINGS,
        )

    @ez.publisher(OUTPUT_SIGNAL)
    async def pub_chunk(self) -> typing.AsyncGenerator:
        for msg in self.STATE.gen:
            # TODO: Direct msg to OUTPUT_TRIGGER if type is SampleTriggerMessage
            yield self.OUTPUT_SIGNAL, msg
            await asyncio.sleep(0)

        ez.logger.debug(f"File ({self.SETTINGS.filepath}) exhausted.")
        if self.SETTINGS.self_terminating:
            raise ez.NormalTermination
        yield self.OUTPUT_TERM, ez.Flag
