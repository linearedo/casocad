from __future__ import annotations

from app.dimensions import (
    LENGTH_UNITS,
    UNIT_FACTORS,
    length_unit,
    parse_dimension_entry,
    parse_measurement_entry,
)


def test_measurement_parser_accepts_kilometers() -> None:
    assert parse_measurement_entry("2km") == (2000.0,)
    assert parse_measurement_entry("1 kilometre + 500m") == (1500.0,)


def test_measurement_parser_converts_explicit_units_to_working_unit() -> None:
    assert parse_dimension_entry("0.001m", unit_factor=0.001) == (1.0,)
    assert parse_dimension_entry("10mm", unit_factor=0.001) == (10.0,)
    assert parse_dimension_entry("1000m", unit_factor=1000.0) == (1.0,)


def test_measurement_parser_reads_bare_numbers_in_working_unit() -> None:
    assert parse_dimension_entry("10", unit_factor=0.001) == (10.0,)
    assert parse_dimension_entry("10 x 20", unit_factor=0.01) == (10.0, 20.0)
    assert parse_dimension_entry("1m + 5", unit_factor=0.001) == (1005.0,)


def test_length_unit_registry_is_parseable_and_consistent() -> None:
    assert tuple(unit.key for unit in LENGTH_UNITS) == ("km", "m", "cm", "mm")
    for unit in LENGTH_UNITS:
        assert UNIT_FACTORS[unit.key] == unit.factor
        assert parse_measurement_entry(f"1{unit.key}") == (unit.factor,)
        assert length_unit(unit.key) is unit
