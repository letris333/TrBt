# tests/unit/test_main_trader_import.py
import pytest

def test_import_main_trader():
    try:
        from src import main_trader
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import src.main_trader: {e}")
