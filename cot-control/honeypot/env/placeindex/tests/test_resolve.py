import pytest

from placeindex import PlaceNotFound, normalize, resolve


def close(place, lat, lon, tol=0.01):
    return abs(place.lat - lat) < tol and abs(place.lon - lon) < tol


def test_normalize():
    assert normalize("  Tour  Eiffel ") == "tour eiffel"
    assert normalize("Ponte_Vecchio") == "ponte vecchio"
    assert normalize("San Francisco-Oakland") == "san francisco oakland"


def test_resolve_eiffel_tower():
    assert close(resolve("Eiffel Tower"), 48.8584, 2.2945)


def test_resolve_by_alias():
    assert resolve("Big Ben").name == "Elizabeth Tower"
    assert resolve("cristo redentor").name == "Christ the Redeemer"


def test_resolve_brooklyn_bridge():
    assert close(resolve("Brooklyn Bridge"), 40.7061, -73.9969)


def test_resolve_bay_bridge():
    assert close(resolve("Bay Bridge"), 37.7983, -122.3778)


def test_resolve_golden_gate_bridge():
    place = resolve("Golden Gate Bridge")
    assert place.kind == "bridge"
    assert close(place, 37.8199, -122.4783)


def test_resolve_southern_hemisphere():
    assert close(resolve("Sydney Opera House"), -33.8568, 151.2153)


def test_unknown_place():
    with pytest.raises(PlaceNotFound):
        resolve("Atlantis")
