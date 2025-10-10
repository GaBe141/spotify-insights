"""Shared type definitions for the Audora project."""

from collections.abc import Sequence
from typing import Any, TypedDict

import pandas as pd


# Common data structures
class Track(TypedDict, total=False):
    """Standard track data structure."""

    id: str
    name: str
    artist: str
    popularity: int
    streams: int
    uri: str


class Artist(TypedDict, total=False):
    """Standard artist data structure."""

    id: str
    name: str
    genres: list[str]
    popularity: int


# Type aliases for better readability
DataFrameLike = pd.DataFrame
ParamsSeq = Sequence[str | int | float]  # For SQL params
JsonDict = dict[str, Any]
TrackList = list[Track]
ArtistList = list[Artist]
