# order_flow_analyzer.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple, Any
from configparser import ConfigParser
from datetime import datetime, timedelta
import traceback

logger = logging.getLogger(__name__)

# Constants for order flow analysis
ABSORPTION_LEVEL_TYPES = ["BID_ABSORPTION", "ASK_ABSORPTION", "NEUTRAL"]
TRADER_TRAP_TYPES = ["BULL_TRAP", "BEAR_TRAP", "NO_TRAP"]

class OrderFlowAnalyzer:
    def __init__(self, config_manager: ConfigParser):
        self.config = config_manager
        self._of_config: Dict[str, Any] = {}
        self._initialize_of_internal()
        logger.info("OrderFlowAnalyzer initialized.")

    def _initialize_of_internal(self):
        """
        Initialise les paramètres de configuration de l'analyse de flux d'ordres.
        """
        self._of_config = {
            'absorption_volume_threshold': self.config.getfloat('order_flow', 'absorption_volume_threshold', fallback=100000),
            'absorption_volume_ratio_min': self.config.getfloat('order_flow', 'absorption_volume_ratio_min', fallback=0.3),
            'large_order_threshold': self.config.getfloat('order_flow', 'large_order_threshold', fallback=10000),
            'price_imbalance_window': self.config.getfloat('order_flow', 'price_imbalance_window', fallback=2.0)
        }
        logger.info(f"Order Flow parameters initialized: {self._of_config}")

    def calculate_cumulative_volume_delta(self, df_of_data: pd.DataFrame) -> float:
        """
        Calcule le Cumulative Volume Delta (CVD) à partir des données de trades.
        CVD est l'agrégation du volume signé: buy volume positif, sell volume négatif.
        """
        if df_of_data is None or df_of_data.empty:
            return 0.0
        
        if not all(col in df_of_data.columns for col in ['volume', 'side']):
            logger.warning("Les données OF manquent des colonnes 'volume' ou 'side' pour le calcul CVD")
            return 0.0
        
        try:
            df_of_data['signed_volume'] = df_of_data.apply(
                lambda row: row['volume'] if row['side'].lower() == 'buy' else -row['volume'], axis=1
            )
            
            cvd = df_of_data['signed_volume'].sum()
            logger.debug(f"CVD calculé: {cvd} sur {len(df_of_data)} trades")
            return float(cvd)
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul CVD: {e}")
            logger.error(traceback.format_exc())
            return 0.0

    def analyze_absorption_levels(self, df_of_data: pd.DataFrame) -> Dict:
        """
        Analyse les niveaux d'absorption dans les données de trades.
        L'absorption se produit lorsqu'un grand volume est négocié à un niveau de prix
        sans que le prix ne bouge significativement, indiquant un acteur fort.
        """
        absorption_info = {
            'absorption_level': 0.0,
            'absorption_strength': 0.0,
            'absorption_type': ABSORPTION_LEVEL_TYPES[2],
            'absorption_volume': 0.0,
            'absorption_volume_ratio': 0.0,
            'absorption_buy_sell_ratio': 0.0
        }
        
        if df_of_data is None or df_of_data.empty:
            return absorption_info
        
        if not all(col in df_of_data.columns for col in ['price', 'volume', 'side']):
            logger.warning("Données manquantes pour l'analyse d'absorption")
            return absorption_info
        
        try:
            absorption_volume_threshold = self._of_config.get('absorption_volume_threshold', 100000)
            absorption_volume_ratio_min = self._of_config.get('absorption_volume_ratio_min', 0.3)
            
            total_volume = df_of_data['volume'].sum()
            
            if total_volume <= 0:
                return absorption_info
            
            volume_by_price = df_of_data.groupby('price')['volume'].sum()
            
            if volume_by_price.empty:
                return absorption_info
                
            max_volume_price = volume_by_price.idxmax()
            max_volume = volume_by_price[max_volume_price]
            
            volume_ratio = max_volume / total_volume
            
            if max_volume > absorption_volume_threshold and volume_ratio > absorption_volume_ratio_min:
                trades_at_level = df_of_data[df_of_data['price'] == max_volume_price]
                
                buy_volume = trades_at_level[trades_at_level['side'].str.lower() == 'buy']['volume'].sum()
                sell_volume = trades_at_level[trades_at_level['side'].str.lower() == 'sell']['volume'].sum()
                
                if buy_volume + sell_volume > 0:
                    buy_sell_ratio = (buy_volume - sell_volume) / (buy_volume + sell_volume)
                else:
                    buy_sell_ratio = 0.0
                
                if buy_sell_ratio > 0.3:
                    absorption_type = ABSORPTION_LEVEL_TYPES[0]
                elif buy_sell_ratio < -0.3:
                    absorption_type = ABSORPTION_LEVEL_TYPES[1]
                else:
                    absorption_type = ABSORPTION_LEVEL_TYPES[2]
                
                absorption_strength = min(1.0, (volume_ratio * 2) * abs(buy_sell_ratio * 1.5))
                
                absorption_info.update({
                    'absorption_level': float(max_volume_price),
                    'absorption_strength': float(absorption_strength),
                    'absorption_type': absorption_type,
                    'absorption_volume': float(max_volume),
                    'absorption_volume_ratio': float(volume_ratio),
                    'absorption_buy_sell_ratio': float(buy_sell_ratio)
                })
            
            return absorption_info
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse d'absorption: {e}")
            logger.error(traceback.format_exc())
            return absorption_info

    def detect_trapped_traders(self, df_of_data: pd.DataFrame, current_price: float, recent_high: float, recent_low: float) -> Dict:
        """
        Détecte les patterns de traders piégés en analysant les mouvements récents des prix.
        """
        trapped_info = {
            'trap_type': TRADER_TRAP_TYPES[2],
            'trap_strength': 0.0,
            'trap_level': 0.0,
            'trap_volume': 0.0,
            'trapped_traders_bias': 0.0
        }
        
        if df_of_data is None or df_of_data.empty:
            return trapped_info
        
        if not all(col in df_of_data.columns for col in ['timestamp', 'price', 'volume']):
            logger.warning("Données manquantes pour la détection de traders piégés")
            return trapped_info
        
        try:
            df_sorted = df_of_data.sort_values('timestamp')
            
            if len(df_sorted) < 10:
                return trapped_info
            
            prices = df_sorted['price'].values
            
            price_range = recent_high - recent_low
            if price_range <= 0:
                return trapped_info
            
            bull_trap_threshold = 0.5 * price_range
            bear_trap_threshold = 0.5 * price_range
            
            if recent_high > prices[-10:-5].max() and current_price < (recent_high - bull_trap_threshold):
                trap_type = TRADER_TRAP_TYPES[0]
                trap_level = recent_high
                
                trap_strength = min(1.0, (recent_high - current_price) / price_range)
                
                peak_time = df_sorted[df_sorted['price'] == recent_high]['timestamp'].max()
                if pd.notna(peak_time):
                    trades_after_peak = df_sorted[df_sorted['timestamp'] > peak_time]
                    trap_volume = trades_after_peak['volume'].sum()
                else:
                    trap_volume = 0.0
                
                trapped_traders_bias = -1.0 * trap_strength
                
                trapped_info.update({
                    'trap_type': trap_type,
                    'trap_strength': float(trap_strength),
                    'trap_level': float(trap_level),
                    'trap_volume': float(trap_volume),
                    'trapped_traders_bias': float(trapped_traders_bias)
                })
                
            elif recent_low < prices[-10:-5].min() and current_price > (recent_low + bear_trap_threshold):
                trap_type = TRADER_TRAP_TYPES[1]
                trap_level = recent_low
                
                trap_strength = min(1.0, (current_price - recent_low) / price_range)
                
                bottom_time = df_sorted[df_sorted['price'] == recent_low]['timestamp'].max()
                if pd.notna(bottom_time):
                    trades_after_bottom = df_sorted[df_sorted['timestamp'] > bottom_time]
                    trap_volume = trades_after_bottom['volume'].sum()
                else:
                    trap_volume = 0.0
                
                trapped_traders_bias = 1.0 * trap_strength
                
                trapped_info.update({
                    'trap_type': trap_type,
                    'trap_strength': float(trap_strength),
                    'trap_level': float(trap_level),
                    'trap_volume': float(trap_volume),
                    'trapped_traders_bias': float(trapped_traders_bias)
                })
            
            return trapped_info
        
        except Exception as e:
            logger.error(f"Erreur lors de la détection des traders piégés: {e}")
            logger.error(traceback.format_exc())
            return trapped_info

    def analyze_large_orders(self, df_of_data: pd.DataFrame) -> Dict:
        """
        Analyse les orders de grande taille pour identifier les acteurs importants du marché.
        """
        large_orders_info = {
            'large_buy_orders_count': 0,
            'large_sell_orders_count': 0,
            'large_buy_volume': 0.0,
            'large_sell_volume': 0.0,
            'large_orders_bias': 0.0
        }
        
        if df_of_data is None or df_of_data.empty:
            return large_orders_info
        
        if not all(col in df_of_data.columns for col in ['volume', 'side']):
            logger.warning("Données manquantes pour l'analyse des ordres larges")
            return large_orders_info
        
        try:
            large_order_threshold = self._of_config.get('large_order_threshold', 10000)
            
            large_orders = df_of_data[df_of_data['volume'] >= large_order_threshold]
            
            if large_orders.empty:
                return large_orders_info
            
            large_buy_orders = large_orders[large_orders['side'].str.lower() == 'buy']
            large_sell_orders = large_orders[large_orders['side'].str.lower() == 'sell']
            
            large_buy_count = len(large_buy_orders)
            large_sell_count = len(large_sell_orders)
            
            large_buy_volume = large_buy_orders['volume'].sum()
            large_sell_volume = large_sell_orders['volume'].sum()
            
            total_large_volume = large_buy_volume + large_sell_volume
            if total_large_volume > 0:
                large_orders_bias = (large_buy_volume - large_sell_volume) / total_large_volume
            else:
                large_orders_bias = 0.0
            
            large_orders_info.update({
                'large_buy_orders_count': large_buy_count,
                'large_sell_orders_count': large_sell_count,
                'large_buy_volume': float(large_buy_volume),
                'large_sell_volume': float(large_sell_volume),
                'large_orders_bias': float(large_orders_bias)
            })
            
            return large_orders_info
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des ordres larges: {e}")
            logger.error(traceback.format_exc())
            return large_orders_info

    def analyze_order_flow_for_period(
        self,
        df_of_data_period: Optional[pd.DataFrame]
    ) -> Dict:
        """
        Analyse les données OF pour une période spécifique pour dériver des features (CVD, Absorption, etc.).
        """
        features_of = {
            'of_cvd': 0.0,
            'of_absorption_score': 0.0,
            'of_trapped_traders_bias': 0.0,
            'of_large_orders_bias': 0.0,
            'of_absorption_level': 0.0,
            'of_absorption_type': ABSORPTION_LEVEL_TYPES[2],
            'of_trap_type': TRADER_TRAP_TYPES[2],
            'of_trap_level': 0.0
        }

        if df_of_data_period is None or df_of_data_period.empty:
            return features_of
        
        try:
            features_of['of_cvd'] = self.calculate_cumulative_volume_delta(df_of_data_period)
            
            absorption_info = self.analyze_absorption_levels(df_of_data_period)
            features_of.update({
                'of_absorption_score': absorption_info['absorption_strength'] * (1 if absorption_info['absorption_type'] == ABSORPTION_LEVEL_TYPES[0] else -1 if absorption_info['absorption_type'] == ABSORPTION_LEVEL_TYPES[1] else 0),
                'of_absorption_level': absorption_info['absorption_level'],
                'of_absorption_type': absorption_info['absorption_type']
            })
            
            if 'price' in df_of_data_period.columns:
                current_price = df_of_data_period['price'].iloc[-1] if not df_of_data_period.empty else 0.0
                recent_high = df_of_data_period['price'].max()
                recent_low = df_of_data_period['price'].min()
                
                trapped_info = self.detect_trapped_traders(df_of_data_period, current_price, recent_high, recent_low)
                features_of.update({
                    'of_trapped_traders_bias': trapped_info['trapped_traders_bias'],
                    'of_trap_type': trapped_info['trap_type'],
                    'of_trap_level': trapped_info['trap_level']
                })
            
            large_orders_info = self.analyze_large_orders(df_of_data_period)
            features_of.update({
                'of_large_orders_bias': large_orders_info['large_orders_bias']
            })
            
            of_composite_score = (
                features_of['of_cvd'] / 100000 +
                features_of['of_absorption_score'] +
                features_of['of_trapped_traders_bias'] +
                features_of['of_large_orders_bias']
            ) / 4
            
            features_of['of_composite_score'] = np.clip(of_composite_score, -1.0, 1.0)
            
            logger.debug(f"Order Flow features calculées: {features_of}")
            return features_of
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse OF: {e}")
            logger.error(traceback.format_exc())
            return features_of

    def fetch_historical_imbalances(
        self,
        df_of_data: Optional[pd.DataFrame],
        price_levels: List[float]
    ) -> Dict[float, float]:
        """
        Identifie les déséquilibres historiques à des niveaux de prix spécifiques.
        """
        imbalances = {}
        
        if df_of_data is None or df_of_data.empty or not price_levels:
            return imbalances
        
        try:
            price_window = self._of_config.get('price_imbalance_window', 2.0)
            
            for level in price_levels:
                price_min = level - price_window/2
                price_max = level + price_window/2
                
                trades_at_level = df_of_data[(df_of_data['price'] >= price_min) & (df_of_data['price'] <= price_max)]
                
                if not trades_at_level.empty:
                    buy_vol = trades_at_level[trades_at_level['side'].str.lower() == 'buy']['volume'].sum()
                    sell_vol = trades_at_level[trades_at_level['side'].str.lower() == 'sell']['volume'].sum()
                    
                    total_vol = buy_vol + sell_vol
                    if total_vol > 0:
                        imbalance = (buy_vol - sell_vol) / total_vol
                        imbalances[level] = imbalance
                    else:
                        imbalances[level] = 0.0
            
            return imbalances
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des déséquilibres historiques: {e}")
            logger.error(traceback.format_exc())
            return imbalances