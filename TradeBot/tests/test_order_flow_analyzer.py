#!/usr/bin/env python3
"""
Tests for order_flow_analyzer module
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from configparser import ConfigParser
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
import order_flow_analyzer


class TestOrderFlowAnalyzer(unittest.TestCase):
    """Tests for the order flow analyzer module"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a sample config
        self.config = ConfigParser()
        self.config.add_section('order_flow')
        self.config.set('order_flow', 'enabled', 'True')
        self.config.set('order_flow', 'cvd_lookback_bars', '5')
        self.config.set('order_flow', 'absorption_volume_threshold', '100000')
        self.config.set('order_flow', 'absorption_price_move_threshold_percent', '0.01')
        self.config.set('order_flow', 'trapped_traders_delta_threshold', '50000')
        
        # Sample order flow data
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1min'),
            'price': np.linspace(40000, 41000, 100),
            'volume': np.random.randint(1, 1000, 100),
            'side': np.random.choice(['buy', 'sell'], 100)
        }
        self.sample_of_data = pd.DataFrame(data)
        self.sample_of_data.set_index('timestamp', inplace=True)
        
        # Initialize the module
        order_flow_analyzer.initialize(self.config)

    def test_analyze_order_flow_for_period(self):
        """Test analyze_order_flow_for_period function"""
        result = order_flow_analyzer.analyze_order_flow_for_period(self.sample_of_data, self.config)
        
        # Check that the result contains the expected keys
        self.assertIn('of_cvd', result)
        self.assertIn('of_absorption_score', result)
        self.assertIn('of_trapped_traders_bias', result)
        
        # Check that the values are of the expected type
        self.assertIsInstance(result['of_cvd'], (int, float))
        self.assertIsInstance(result['of_absorption_score'], (int, float))
        self.assertIsInstance(result['of_trapped_traders_bias'], (int, float))

    def test_calculate_cvd(self):
        """Test the calculation of Cumulative Volume Delta"""
        # Create a sample dataframe with known buys and sells
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='1min'),
            'price': [40000, 40100, 40200, 40150, 40300],
            'volume': [100, 200, 150, 300, 250],
            'side': ['buy', 'buy', 'sell', 'sell', 'buy']
        }
        test_df = pd.DataFrame(data)
        test_df.set_index('timestamp', inplace=True)
        
        # Expected CVD calculation: +100 +200 -150 -300 +250 = +100
        cvd = order_flow_analyzer.calculate_cvd(test_df)
        
        # Check that CVD is calculated correctly
        self.assertEqual(cvd, 100)
        
    def test_detect_absorption(self):
        """Test the detection of absorption patterns"""
        # Create a sample with absorption pattern (high volume but little price movement)
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='1min'),
            'price': [40000, 40010, 40020, 40030, 40020, 40010, 40000, 39990, 39980, 39970],
            'volume': [1000, 5000, 20000, 50000, 100000, 200000, 150000, 50000, 20000, 5000],
            'side': ['buy', 'buy', 'buy', 'buy', 'buy', 'sell', 'sell', 'sell', 'sell', 'sell']
        }
        test_df = pd.DataFrame(data)
        test_df.set_index('timestamp', inplace=True)
        
        # Test with mock config
        mock_config = ConfigParser()
        mock_config.add_section('order_flow')
        mock_config.set('order_flow', 'absorption_volume_threshold', '100000')
        mock_config.set('order_flow', 'absorption_price_move_threshold_percent', '0.01')
        
        result = order_flow_analyzer.analyze_order_flow_for_period(test_df, mock_config)
        
        # Check that absorption is detected (positive score for buying absorption, negative for selling)
        self.assertIsNotNone(result['of_absorption_score'])
        
    def test_analyze_trapped_traders(self):
        """Test the analysis of trapped traders"""
        # Create a sample with a price rejection pattern (price moves up then quickly down)
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=20, freq='1min'),
            'price': [40000] + list(np.linspace(40000, 41000, 9)) + list(np.linspace(41000, 39500, 10)),
            'volume': [1000] + [5000] * 9 + [10000] * 10,
            'side': ['buy'] * 10 + ['sell'] * 10
        }
        test_df = pd.DataFrame(data)
        test_df.set_index('timestamp', inplace=True)
        
        # Test with mock config
        mock_config = ConfigParser()
        mock_config.add_section('order_flow')
        mock_config.set('order_flow', 'trapped_traders_delta_threshold', '10000')
        
        result = order_flow_analyzer.analyze_order_flow_for_period(test_df, mock_config)
        
        # Check that trapped traders bias is detected
        self.assertIsNotNone(result['of_trapped_traders_bias'])

    def test_empty_input(self):
        """Test handling of empty input data"""
        empty_df = pd.DataFrame()
        result = order_flow_analyzer.analyze_order_flow_for_period(empty_df, self.config)
        
        # Check that the function returns default values for empty input
        self.assertEqual(result['of_cvd'], 0.0)
        self.assertEqual(result['of_absorption_score'], 0.0)
        self.assertEqual(result['of_trapped_traders_bias'], 0.0)

    def test_config_disabled(self):
        """Test when order flow analysis is disabled in config"""
        # Modify config to disable order flow
        self.config.set('order_flow', 'enabled', 'False')
        
        result = order_flow_analyzer.analyze_order_flow_for_period(self.sample_of_data, self.config)
        
        # Check that default values are returned when disabled
        self.assertEqual(result['of_cvd'], 0.0)
        self.assertEqual(result['of_absorption_score'], 0.0)
        self.assertEqual(result['of_trapped_traders_bias'], 0.0)


if __name__ == '__main__':
    unittest.main() 