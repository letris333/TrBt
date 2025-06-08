#!/usr/bin/env python3
"""
Tests for position_manager module
"""

import pytest
from unittest.mock import patch, MagicMock
import json
import tempfile
from configparser import ConfigParser
import pandas as pd
import time

# Add parent directory to path to import modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from src.core.position_manager import PositionManager

class TestPositionManager:
    """Tests for the position manager module"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create a mock config
        self.mock_config_manager = MagicMock()
        self.mock_notification_service = MagicMock()
        
        # Simuler la configuration que PositionManager pourrait attendre
        self.mock_config_manager.get_trading_config.return_value = {
            'max_closed_positions_history': 100 # Exemple
        }
        
        self.position_manager = PositionManager(
            config_manager=self.mock_config_manager,
            notification_service=self.mock_notification_service
        )
        # Assurez-vous que PositionManager a un attribut logger ou mockez-le
        if not hasattr(self.position_manager, 'logger'):
            self.position_manager.logger = MagicMock()

    def test_open_new_long_position(self):
        """Teste l'ouverture d'une nouvelle position longue."""
        timestamp = time.time()
        self.position_manager.open_position(
            symbol_market='BTC/USDT',
            side='buy',
            amount=1.0,
            entry_price=30000.0,
            entry_timestamp=timestamp,
            stop_loss_price=29000.0,
            take_profit_price=31000.0,
            trade_id='trade_001'
        )
        position = self.position_manager.get_position('BTC/USDT')
        assert position is not None
        assert position['side'] == 'buy'
        assert position['amount'] == 1.0
        assert position['trade_id'] == 'trade_001'

    def test_close_existing_position(self):
        """Teste la fermeture d'une position existante."""
        open_timestamp = time.time() - 3600 # 1 heure avant
        self.position_manager.open_position('ETH/USDT', 'buy', 10.0, 2000.0, open_timestamp, trade_id='trade_002')
        
        close_timestamp = time.time()
        closed_position_summary = self.position_manager.close_position(
            symbol_market='ETH/USDT',
            exit_price=2100.0,
            exit_timestamp=close_timestamp,
            reason="Take Profit"
        )
        assert self.position_manager.get_position('ETH/USDT') is None
        assert closed_position_summary is not None
        assert closed_position_summary['symbol_market'] == 'ETH/USDT'
        assert pytest.approx(closed_position_summary['pnl_usd'], (2100.0 - 2000.0) * 10.0) 
        assert closed_position_summary['pnl_percent'] > 0 # Profit

    def test_update_pnl_long_position_profit(self):
        """Teste la mise à jour du P&L pour une position longue en profit."""
        timestamp = time.time()
        self.position_manager.open_position('BTC/USDT', 'buy', 1.0, 30000.0, timestamp, trade_id='trade_003')
        self.position_manager.update_position_pnl('BTC/USDT', 31000.0)
        position = self.position_manager.get_position('BTC/USDT')
        assert position is not None
        assert pytest.approx(position['unrealized_pnl_usd'], (31000.0 - 30000.0) * 1.0)
        assert position['unrealized_pnl_percent'] > 0

    def test_update_pnl_short_position_loss(self):
        """Teste la mise à jour du P&L pour une position courte en perte."""
        timestamp = time.time()
        # Assurez-vous que votre PositionManager gère correctement les positions 'sell'
        self.position_manager.open_position('BTC/USDT', 'sell', 1.0, 30000.0, timestamp, trade_id='trade_004')
        self.position_manager.update_position_pnl('BTC/USDT', 31000.0) # Prix monte, donc perte pour short
        position = self.position_manager.get_position('BTC/USDT')
        assert position is not None
        assert pytest.approx(position['unrealized_pnl_usd'], (30000.0 - 31000.0) * 1.0) # (entry - current) * amount for short
        assert position['unrealized_pnl_percent'] < 0

    def test_get_all_open_positions(self):
        """Teste la récupération de toutes les positions ouvertes."""
        ts1 = time.time()
        ts2 = time.time() + 10
        self.position_manager.open_position('BTC/USDT', 'buy', 1, 30000, ts1, trade_id='t1')
        self.position_manager.open_position('ETH/USDT', 'sell', 10, 2000, ts2, trade_id='t2')
        open_positions = self.position_manager.get_all_open_positions()
        assert len(open_positions) == 2
        assert 'BTC/USDT' in open_positions
        assert 'ETH/USDT' in open_positions

if __name__ == '__main__':
    pytest.main()