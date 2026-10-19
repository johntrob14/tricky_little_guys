# Gazetteer data format

`placeindex/data/landmarks.csv` is UTF-8 CSV with a header row:

| column | type | notes |
|---|---|---|
| `name` | str | canonical name |
| `kind` | str | `bridge`, `tower`, `building`, `monument`, `park`, ... |
| `lat` | float | WGS84 latitude, -90 to 90 |
| `lon` | float | WGS84 longitude, -180 to 180 |
| `aliases` | str | `;`-separated alternative names, may be empty |

Rows whose coordinates are out of range are skipped at load time (logged at DEBUG level on the
`placeindex.gazetteer` logger). The refresh script exports `lat,lon` in that order; check the column order
when importing from other sources, some of which use `lon,lat`.
