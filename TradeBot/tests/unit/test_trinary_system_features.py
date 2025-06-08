import sys
from unittest.mock import MagicMock

# Mock potentially problematic libraries BEFORE they are imported by any application code
sys.modules['ccxt'] = MagicMock()
sys.modules['smtplib'] = MagicMock()
sys.modules['requests'] = MagicMock()

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.pipeline.feature_engineer import build_complete_raw_features
from src.main_trader import HistoricalDataManager, ConfigManager
from src.indicators import calculate_ift, calculate_rdvu, derive_volatility_cycle_position, DEFAULT_TRINARY_RSI_PERIOD, DEFAULT_TRINARY_ATR_PERIOD_STABILITY, DEFAULT_TRINARY_ATR_PERIOD_VOLATILITY, DEFAULT_TRINARY_ATR_SMA_PERIOD, PI


@pytest.fixture
def sample_ohlcv_data():
    """Provides a small DataFrame of OHLCV data for testing."""
    data = {
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        'sentiment_score': [0.5, -0.2, 0.8, 0.1, -0.6, 0.3, 0.0, 0.9, -0.1, 0.7]
    }
    index = pd.date_range(start='2023-01-01', periods=10, freq='D')
    return pd.DataFrame(data, index=index)

@pytest.fixture
def sample_ta_features_data(sample_ohlcv_data):
    """Provides sample TA features corresponding to sample_ohlcv_data."""
    df = pd.DataFrame(index=sample_ohlcv_data.index)
    df['RSI'] = [50, 60, 70, 55, 40, 30, 45, 65, 75, 50] # Generic RSI
    df[f'RSI_{DEFAULT_TRINARY_RSI_PERIOD}'] = df['RSI'] # Specific RSI for testing
    df['ATR'] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9] # Generic ATR
    df[f'ATR_{DEFAULT_TRINARY_ATR_PERIOD_STABILITY}'] = df['ATR'] # Specific ATR for stability
    
    # Corrected column name for ATR Trinary Volatility
    atr_vol_col = f'ATR_TRINARY_VOLATILITY_{DEFAULT_TRINARY_ATR_PERIOD_VOLATILITY}'
    df[atr_vol_col] = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    
    # Corrected column name for SMA of ATR Trinary Volatility
    atr_sma_vol_col = f'ATR_SMA_TRINARY_VOLATILITY_{DEFAULT_TRINARY_ATR_PERIOD_VOLATILITY}_{DEFAULT_TRINARY_ATR_SMA_PERIOD}'
    df[atr_sma_vol_col] = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45]
    
    df['BB_WIDTH'] = [0.02, 0.022, 0.021, 0.023, 0.025, 0.024, 0.026, 0.023, 0.022, 0.025]
    return df

@pytest.fixture
def default_trinary_config():
    """Provides default Trinary system configuration."""
    return {
        'trinary_rsi_period': DEFAULT_TRINARY_RSI_PERIOD,
        'trinary_atr_period_stability': DEFAULT_TRINARY_ATR_PERIOD_STABILITY,
        'trinary_atr_period_volatility': DEFAULT_TRINARY_ATR_PERIOD_VOLATILITY,
        'trinary_atr_sma_period': DEFAULT_TRINARY_ATR_SMA_PERIOD,
        'trinary_bbw_percentile_window': 5 # Small window for testing
    }

@pytest.fixture
def mock_config_manager(default_trinary_config):
    """Mocks ConfigManager to return specific trinary_config."""
    mock_cm = MagicMock(spec=ConfigManager)
    mock_cm.get_section.return_value = default_trinary_config
    # Mock getint for trinary_bbw_percentile_window specifically for HistoricalDataManager
    def getint_side_effect(section, key, fallback=None):
        if section == 'trinary_config' and key == 'trinary_bbw_percentile_window':
            return default_trinary_config.get(key, fallback)
        # For other calls, return the fallback or a default like 0 or 1 if no fallback.
        # This prevents errors if HistoricalDataManager tries to get other int configs.
        if fallback is not None:
            return fallback
        # If ConfigManager.getint is called for other keys without a fallback,
        # we might need to provide specific mocks or a generic default.
        # For now, let's return a common default like 0 and print a warning.
        print(f"PYTEST_DEBUG: Unexpected getint call in mock_config_manager: section='{section}', key='{key}'. Returning default 0.")
        return 0
    mock_cm.getint.side_effect = getint_side_effect
    return mock_cm


class TestTrinarySystemFeatures:
    def test_build_raw_features_ift_components(
        self, sample_ohlcv_data, sample_ta_features_data, default_trinary_config
    ):
        """Test R_IFT component calculations in build_complete_raw_features."""
        ohlcv_row = sample_ohlcv_data.iloc[5]
        ta_features_row = sample_ta_features_data.iloc[5]
        pi_ratings = {'R_H': 0.6, 'R_A': 0.4}
        ms_features = {}
        of_features = {}

        features = build_complete_raw_features(
            ohlcv_row_series=ohlcv_row,
            ms_features=ms_features,
            of_features=of_features,
            pi_ratings=pi_ratings,
            ta_features_row=ta_features_row,
            indicator_config=default_trinary_config
        )

        # Expected market_momentum: (RSI - 50) / 50 = (30 - 50) / 50 = -0.4
        assert np.isclose(features['market_momentum'], -0.4)

        # Expected stability_factor: close / ATR_stability = 105.5 / 1.5
        expected_stability_factor = 105.5 / (1.5 + 1e-9)
        assert np.isclose(features['stability_factor_ift_input'], expected_stability_factor)
        
        # Expected R_IFT (manual calculation for this specific case)
        exp_ift_trend, exp_ift_strength = calculate_ift(0.6, 0.4, -0.4, expected_stability_factor)
        assert np.isclose(features['R_IFT_Trend'], exp_ift_trend)
        assert np.isclose(features['R_IFT_Strength'], exp_ift_strength)

    def test_build_raw_features_rdvu_components(
        self, sample_ohlcv_data, sample_ta_features_data, default_trinary_config
    ):
        """Test R_DVU component preparations in build_complete_raw_features."""
        ohlcv_row = sample_ohlcv_data.iloc[5] # sentiment_score = 0.3
        ta_features_row = sample_ta_features_data.iloc[5]
        # ATR_volatility = 1.6, ATR_SMA_volatility = 1.25
        pi_ratings = {}
        ms_features = {}
        of_features = {}

        features = build_complete_raw_features(
            ohlcv_row_series=ohlcv_row,
            ms_features=ms_features,
            of_features=of_features,
            pi_ratings=pi_ratings,
            ta_features_row=ta_features_row,
            indicator_config=default_trinary_config
        )

        # Expected base_stability_index: abs(sentiment_score) = abs(0.3) = 0.3
        assert np.isclose(features['base_stability_index_rdvu_input'], 0.3)
        assert np.isclose(features['atr_current_rdvu_input'], 1.6)
        assert np.isclose(features['atr_average_rdvu_input'], 1.25)

    def test_historical_data_manager_rdvu_calculation(
        self, sample_ohlcv_data, sample_ta_features_data, mock_config_manager, default_trinary_config
    ):
        """Test end-to-end R_DVU calculation in HistoricalDataManager."""
        symbol = "TEST/USD"
        
        # Mock HistoricalDataManager's dependencies
        mock_im = MagicMock()
        mock_im.calculate_rdvu.side_effect = lambda atr_current, atr_average, volatility_cycle_position, base_stability_index, pi_factor=None, cycle_amplitude=None: \
            calculate_rdvu(atr_current, atr_average, volatility_cycle_position, base_stability_index, pi_factor, cycle_amplitude)
        # Add mock for derive_volatility_cycle_position to call the real function
        mock_im.derive_volatility_cycle_position.side_effect = lambda percentile_rank: \
            derive_volatility_cycle_position(percentile_rank)

        mock_fe = MagicMock()
        # Let build_complete_raw_features run mostly as is, but ensure it uses our specific config
        def build_features_side_effect(ohlcv_row_series, ms_features, of_features, pi_ratings, ta_features_row, indicator_config):
            # Call the actual function with the provided indicator_config
            return build_complete_raw_features(ohlcv_row_series, ms_features, of_features, pi_ratings, ta_features_row, indicator_config)
        mock_fe.build_complete_raw_features.side_effect = build_features_side_effect

        hdm = HistoricalDataManager(
            config_manager=mock_config_manager, 
            indicators_manager=mock_im, 
            feature_engineer=mock_fe,
            max_history_length=100
        )

        # Initialize storage for the symbol
        hdm.historical_ohlcv[symbol] = sample_ohlcv_data
        hdm.ta_features_history[symbol] = sample_ta_features_data
        hdm.ms_features_history[symbol] = {idx: {} for idx in sample_ohlcv_data.index}
        hdm.of_features_history[symbol] = {idx: {} for idx in sample_ohlcv_data.index}
        hdm.complete_features_history[symbol] = pd.DataFrame() # Start empty

        # Mock get_ratings from indicators_manager
        mock_im.get_ratings.return_value = {'R_H': 0.5, 'R_A': 0.5} # Neutral Pi-Ratings for simplicity

        # Run the update process
        hdm.update_complete_features(symbol)

        complete_features = hdm.complete_features_history[symbol]

        assert 'R_DVU' in complete_features.columns
        assert not complete_features['R_DVU'].isnull().all() # Ensure R_DVU is calculated

        # Check a specific row (e.g., iloc[5], matching other tests)
        # For iloc[5]:
        # ohlcv_row: close=105.5, sentiment_score=0.3 -> base_stability_index_rdvu_input = 0.3
        # ta_features_row: ATR_volatility=1.6, ATR_SMA_volatility=1.25, BB_WIDTH=0.024
        # RSI=30 -> market_momentum = -0.4
        # ATR_stability=1.5 -> stability_factor_ift_input = 105.5/1.5
        
        # volatility_cycle_pos for iloc[5] (index 2023-01-06)
        # BB_WIDTH values: [0.02, 0.022, 0.021, 0.023, 0.025, 0.024, ...]
        # Window = 5, min_periods = 2
        # Ranks (pct=True):
        # iloc[0]: NaN (or 0.5 if min_p=1)
        # iloc[1]: 1.0 (0.022 > 0.02)
        # iloc[2]: 0.333... (0.021 is smallest of [0.02, 0.022, 0.021])
        # iloc[3]: 0.75 (0.023 is 3rd of [0.02, 0.022, 0.021, 0.023])
        # iloc[4]: 1.0 (0.025 is largest of [0.02, 0.022, 0.021, 0.023, 0.025])
        # iloc[5]: 0.8 (0.024 is 3rd of [0.022,0.021,0.023,0.025,0.024] after sorting [0.021,0.022,0.023,0.024,0.025]) -> rank is (2+1)/5 = 0.6
        # volatility_cycle_pos = 0.6 * 2 = 1.2

        # Check iloc[5] which has index '2023-01-06'
        row_to_check = complete_features.loc['2023-01-06']
        expected_percentile_rank_iloc5 = 0.8 
        expected_volatility_cycle_pos_iloc5 = expected_percentile_rank_iloc5 * 2 * PI
        assert np.isclose(row_to_check['volatility_cycle_pos'], expected_volatility_cycle_pos_iloc5)
        
        expected_rdvu = calculate_rdvu(
            atr_current=row_to_check['atr_current_rdvu_input'], # Should be 1.6 from fixture
            atr_average=row_to_check['atr_average_rdvu_input'], # Should be 1.25 from fixture
            volatility_cycle_position=row_to_check['volatility_cycle_pos'], # now 0.8 * 2 * PI
            base_stability_index=row_to_check['base_stability_index_rdvu_input'] # Should be 0.3 from fixture
        )
        assert np.isclose(row_to_check['R_DVU'], expected_rdvu)
        
        # Check that R_IFT features are also present
        assert 'R_IFT_Trend' in complete_features.columns
        assert 'R_IFT_Strength' in complete_features.columns
        assert not complete_features['R_IFT_Trend'].isnull().all()

# To run these tests: pytest tests/unit/test_trinary_system_features.py
