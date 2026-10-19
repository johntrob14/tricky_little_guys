# placeindex

Resolve landmark and place names to coordinates from a bundled gazetteer.

```python
>>> from placeindex import resolve
>>> resolve("Eiffel Tower")
Place(name='Eiffel Tower', kind='tower', lat=48.8584, lon=2.2945, aliases=('Tour Eiffel',))
```

Names are matched case-insensitively, ignoring punctuation and repeated whitespace. Each place can also be
looked up by any of its aliases.

## Data

The gazetteer lives in `placeindex/data/landmarks.csv`. See `docs/data-format.md` for the column format and
`CHANGELOG.md` for refresh history.

## Development

    python -m pytest -q
