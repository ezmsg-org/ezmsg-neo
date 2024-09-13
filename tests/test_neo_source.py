import os
from pathlib import Path

from ezmsg.neo.source import NeoIterator, NeoIteratorSettings

"""
def test_brainvision_playback():

    source_path = None  # TODO - web source.
    settings = NeoIteratorSettings(filepath=source_path)
    neo_iter = NeoIterator(settings)

    data_msg_count = 0
    for msg in neo_iter:
        if msg.key == "events":
            print(msg)
        else:
            data_msg_count += 1

    print(f"Data messages: {data_msg_count}")
"""