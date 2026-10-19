"""Gazetteer loading and name lookup."""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .errors import PlaceNotFound

log = logging.getLogger(__name__)

DATA = Path(__file__).parent / "data" / "landmarks.csv"
_SEP = re.compile(r"[\W_]+")


@dataclass(frozen=True)
class Place:
    name: str
    kind: str
    lat: float
    lon: float
    aliases: tuple[str, ...] = ()


def normalize(name: str) -> str:
    """Lowercase, and collapse punctuation, underscores and whitespace to single spaces."""
    return _SEP.sub(" ", name).strip().casefold()


def _in_range(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def load(path: Path | str = DATA) -> dict[str, Place]:
    """Index of normalized name and alias -> Place. Rows with out-of-range coordinates are skipped."""
    index: dict[str, Place] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lat, lon = float(row["lat"]), float(row["lon"])
            if not _in_range(lat, lon):
                log.debug("skipping %r: coordinates out of range (%s, %s)", row["name"], lat, lon)
                continue
            aliases = tuple(a.strip() for a in row["aliases"].split(";") if a.strip())
            place = Place(row["name"], row["kind"], lat, lon, aliases)
            for key in (row["name"], *aliases):
                index.setdefault(normalize(key), place)
    return index


@lru_cache(maxsize=1)
def _default_index() -> dict[str, Place]:
    return load()


def resolve(name: str) -> Place:
    """The place whose name or alias matches `name`."""
    try:
        return _default_index()[normalize(name)]
    except KeyError:
        raise PlaceNotFound(f"no place matching {name!r}") from None
