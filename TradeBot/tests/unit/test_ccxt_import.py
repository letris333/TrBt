# tests/unit/test_ccxt_import.py
import pytest

def test_import_ccxt():
    try:
        import ccxt
        assert True
        print("ccxt imported successfully") # Add a print statement
    except ImportError as e:
        pytest.fail(f"Failed to import ccxt: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during ccxt import: {e}")
