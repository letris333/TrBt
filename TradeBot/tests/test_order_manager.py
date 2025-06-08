import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

# Importer la vraie classe OrderManager et ConfigManager (si nécessaire)
from src.core.order_manager import OrderManager
# from src.core.config_manager import ConfigManager # Décommentez si OrderManager l'attend directement
# from src.core.drm_manager import DRMManager # Si nécessaire

class TestOrderManager(unittest.TestCase):

    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.mock_exchange_client = MagicMock()
        self.mock_config_manager = MagicMock() # Sera passé à OrderManager
        self.mock_drm_manager = MagicMock()    # Sera passé à OrderManager
        
        # Simuler la configuration nécessaire que OrderManager pourrait attendre de ConfigManager
        # Ceci est un exemple, adaptez selon ce que OrderManager utilise réellement de config_manager
        self.mock_config_manager.get_active_exchange_client.return_value = self.mock_exchange_client
        self.mock_config_manager.get_trading_config.return_value = {
            'order_params': {'timeInForce': 'GTC'}, # Exemple de paramètre
            'max_api_retries': 3,
            'api_retry_delay': 1
        }
        # Si DRMManager est un objet complexe, mockez ses méthodes aussi
        self.mock_drm_manager.validate_order_size.return_value = True # Exemple
        
        self.order_manager = OrderManager(
            # exchange_client est maintenant obtenu via config_manager dans beaucoup de vos classes
            # Adaptez la création de OrderManager selon sa vraie signature __init__
            # Si OrderManager prend config_manager et en tire l'exchange_client:
            config_manager=self.mock_config_manager,
            drm_manager=self.mock_drm_manager 
            # Si OrderManager prend exchange_client directement:
            # exchange_client=self.mock_exchange_client,
            # config_manager=self.mock_config_manager, 
            # drm_manager=self.mock_drm_manager
        )
        # Assurez-vous que OrderManager a un attribut logger ou mockez-le si nécessaire
        if not hasattr(self.order_manager, 'logger'):
             self.order_manager.logger = MagicMock()

    def test_create_limit_buy_order(self):
        """Teste la création d'un ordre d'achat limite."""
        self.mock_exchange_client.create_order.return_value = {'id': '123', 'status': 'open', 'symbol': 'BTC/USDT', 'type': 'limit', 'side': 'buy', 'amount': 1.0, 'price': 30000.0}
        # Assurez-vous que drm_manager est correctement mocké si create_order l'utilise
        # self.mock_drm_manager.validate_order_size.return_value = True # Déjà dans setUp

        # La méthode _api_call_with_retry est dans TradingBot, OrderManager utilise directement l'exchange_client
        # ou a sa propre logique de retry. Ici on mock directement create_order de l'exchange.

        order = self.order_manager.create_order(
            symbol_market='BTC/USDT',
            order_type='limit',
            side='buy',
            amount=1.0,
            price=30000.0
        )
        self.assertIsNotNone(order)
        self.assertEqual(order['id'], '123')
        # Adaptez l'appel attendu selon la vraie signature de create_order dans ccxt ou votre wrapper
        self.mock_exchange_client.create_order.assert_called_once_with(
            symbol='BTC/USDT', type='limit', side='buy', amount=1.0, price=30000.0, params={'timeInForce': 'GTC'}
        )

    def test_create_market_sell_order(self):
        """Teste la création d'un ordre de vente au marché."""
        self.mock_exchange_client.create_order.return_value = {'id': '124', 'status': 'closed', 'symbol': 'BTC/USDT', 'type': 'market', 'side': 'sell', 'amount': 0.5}
        # self.mock_drm_manager.validate_order_size.return_value = True # Déjà dans setUp

        order = self.order_manager.create_order(
            symbol_market='BTC/USDT',
            order_type='market',
            side='sell',
            amount=0.5
        )
        self.assertIsNotNone(order)
        self.assertEqual(order['id'], '124')
        self.mock_exchange_client.create_order.assert_called_once_with(
           symbol='BTC/USDT', type='market', side='sell', amount=0.5, params={'timeInForce': 'GTC'}
        )

    def test_cancel_order(self):
        """Teste l'annulation d'un ordre."""
        self.mock_exchange_client.cancel_order.return_value = {'id': '123', 'status': 'canceled'}
        result = self.order_manager.cancel_order(order_id='123', symbol_market='BTC/USDT')
        self.assertIsNotNone(result)
        # Adaptez l'appel attendu
        self.mock_exchange_client.cancel_order.assert_called_once_with('123', 'BTC/USDT')

    def test_fetch_order_status(self):
        """Teste la récupération du statut d'un ordre."""
        self.mock_exchange_client.fetch_order.return_value = {'id': '123', 'status': 'closed', 'filled': 1.0, 'remaining': 0.0}
        status = self.order_manager.fetch_order_status(order_id='123', symbol_market='BTC/USDT')
        self.assertIsNotNone(status)
        self.assertEqual(status['status'], 'closed')
        self.mock_exchange_client.fetch_order.assert_called_once_with('123', 'BTC/USDT')

    # Ajoutez d'autres tests pour la gestion des erreurs, les nouvelles tentatives, etc.

if __name__ == '__main__':
    unittest.main()
