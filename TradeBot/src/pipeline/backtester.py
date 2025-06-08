# trading_analyzer.py (Début du fichier - ajouts/modifications en premier)

import pandas as pd
import numpy as np
import logging
import time
import sys # For stdout in setup_logging
from datetime import datetime, timedelta, timezone
from configparser import ConfigParser, NoSectionError, NoOptionError
from typing import Dict, Optional, List, Tuple, Any # Ajout de Any
from dataclasses import dataclass, field # Ajout pour BacktestPosition
from ..indicators import calculate_atr

# --- Imports pour Analysis and Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm # Use tqdm.auto for notebook compatibility
import random # For parameter perturbation
import xgboost # Ajouté pour dmatrix dans la partie Backtester ML

# --- Import necessary modules from your project ---
import indicators
import feature_engineer
import model_xgboost # Chargé via _load_models, utilisé par Backtester
import model_futurequant # Chargé via _load_models, utilisé par Backtester
import strategy_hybrid
import market_structure_analyzer
import order_flow_analyzer
import training_pipeline

# --- Configuration and Logging ---
CONFIG_FILE = 'config.ini'
logger = logging.getLogger(__name__)
_prepared_data_global = None

# ... (setup_logging, get_config_option, get_config_list_lower, get_strat_param restent inchangés)
# ... (calculate_fib_tp_sl_prices reste inchangé)

# --- NOUVELLE DATACLASS BacktestPosition (de input_file_2.py) ---
@dataclass
class BacktestPosition:
    symbol_market: str
    entry_price: float
    qty: float
    side: str  # 'long' or 'short'
    tp_price: Optional[float]
    sl_price: Optional[float]
    entry_timestamp: pd.Timestamp
    # entry_bar_index: int # Moins crucial si on se base sur les timestamps, mais peut être utile

    setup_quality: str = 'NONE'
    initial_risk_multiplier: float = 1.0
    
    # Pour la gestion avancée des sorties
    moved_to_be: bool = False
    trailing_sl_active: bool = False
    trailing_sl_price: Optional[float] = None # Prix actuel du trailing SL
    highest_price_since_entry: float = 0.0 # Pour positions longues
    lowest_price_since_entry: float = float('inf') # Pour positions courtes
    
    # Pour stocker les détails de sortie
    exit_price: Optional[float] = None
    exit_timestamp: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None # e.g., 'TP', 'SL', 'Signal', 'EndOfData'
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    fee: float = 0.0 # Cumul des frais pour ce trade (ouverture + clôture)

    def __post_init__(self):
        if self.side == 'long':
            self.highest_price_since_entry = self.entry_price
        elif self.side == 'short':
            self.lowest_price_since_entry = self.entry_price

# --- Backtester Core (Fusion de TA_BT et B_BT) ---

class Backtester:
    def __init__(self,
                 prepared_data: Dict[str, pd.DataFrame],
                 config: ConfigParser,
                 strategy_params: Dict[str, Any], # Strategy-specific parameters
                 loaded_models: Dict[str, Any]
                 ):
        if not prepared_data: raise ValueError("prepared_data dictionary is empty.")
        self.prepared_data = prepared_data
        self.config = config
        self.strategy_params = strategy_params
        self.loaded_models = loaded_models

        self.assets_to_backtest: List[str] = list(prepared_data.keys())

        # Portfolio and results (inspiré de B_BT)
        self.initial_capital = strategy_hybrid.get_strat_param( # Utilisation de get_strat_param de strategy_hybrid
            self.strategy_params, 'initial_capital',
            self.config.getfloat('backtester', 'initial_capital', fallback=10000.0),
            value_type=float
        )
        # Utiliser taker_fee pour les simulations car les ordres marché/stop sont généralement des taker
        self.fee_percent = strategy_hybrid.get_strat_param(
            self.strategy_params, 'transaction_fee_percent',
            self.config.getfloat('backtester', 'taker_fee', fallback=0.001), # Utilisation de taker_fee
            value_type=float
        )
        self.slippage_percent = strategy_hybrid.get_strat_param(
             self.strategy_params, 'slippage_percent',
             self.config.getfloat('backtester', 'slippage_percent', fallback=0.0005),
             value_type=float
        )


        self.cash: float = self.initial_capital
        self.open_positions: List[BacktestPosition] = []
        self.closed_trades: List[BacktestPosition] = []
        self.daily_portfolio_history: List[Dict[str, Any]] = []

        # Load ML components and training params (de TA_BT)
        self.xgb_model = loaded_models.get('xgb') # Nom de clé 'xgb' utilisé dans _load_models
        self.fq_model = loaded_models.get('fq')   # Nom de clé 'fq' utilisé dans _load_models
        self.feature_scaler = loaded_models.get('scaler')
        training_params_from_load = loaded_models.get('params', {}) # 'params' est la clé dans _load_models
        
        self.xgb_feature_names: List[str] = training_params_from_load.get('xgboost_features', [])
        self.fq_feature_names: List[str] = training_params_from_load.get('feature_columns', []) # B_BT utilise 'all_potential_features', TA_BT 'feature_columns'
        self.fq_quantiles = training_params_from_load.get('futurequant_quantiles', [])
        self.all_feature_columns = training_params_from_load.get('feature_columns', [])


        self.ml_ready = True
        if self.xgb_model is None or self.feature_scaler is None or not self.xgb_feature_names or not self.all_feature_columns:
             logger.warning("XGBoost model, scaler ou feature names manquants. XGBoost prediction sera limitée.")
             if self.xgb_model is None: self.ml_ready = False # XGBoost est plus critique
        if self.fq_model is None or not self.fq_quantiles:
             logger.warning("FutureQuant model ou quantiles manquants. FQ prediction sera limitée.")
             # self.ml_ready reste True si XGBoost est OK

        # Column names from config (commun aux deux, TA_BT le faisait déjà bien)
        self.ts_col = get_config_option(config, 'backtest', 'timestamp_col', fallback='timestamp')
        self.open_col = get_config_option(config, 'backtest', 'open_col', fallback='open')
        self.high_col = get_config_option(config, 'backtest', 'high_col', fallback='high')
        self.low_col = get_config_option(config, 'backtest', 'low_col', fallback='low')
        self.close_col = get_config_option(config, 'backtest', 'close_col', fallback='close')
        self.volume_col = get_config_option(config, 'backtest', 'volume_col', fallback='volume')
        self.rh_col = get_config_option(config, 'backtest', 'rh_col', fallback='R_H') # Pour Pi-Ratings
        self.ra_col = get_config_option(config, 'backtest', 'ra_col', fallback='R_A') # Pour Pi-Ratings
        self.sentiment_col = get_config_option(config, 'backtest', 'sentiment_col', fallback='sentiment_score')

        # Max concurrent positions (de TA_BT)
        self.max_concurrent_positions = get_config_option(config, 'trading', 'max_concurrent_positions', fallback=5, value_type=int)

        # DRM parameters (de TA_BT et B_BT, consolidé)
        # Utilisera get_strat_param de strategy_hybrid pour lire depuis strategy_params ou config
        self.move_to_be_profit_percent = strategy_hybrid.get_strat_param(self.strategy_params, 'move_to_be_profit_percent', 0.005, float, config_parser=self.config, section_config='dynamic_risk')
        self.trailing_stop_profit_percent_start = strategy_hybrid.get_strat_param(self.strategy_params, 'trailing_stop_profit_percent_start', 0.01, float, config_parser=self.config, section_config='dynamic_risk')
        self.trailing_stop_distance_percent = strategy_hybrid.get_strat_param(self.strategy_params, 'trailing_stop_distance_percent', 0.005, float, config_parser=self.config, section_config='dynamic_risk')
        
        daily_pnl_adj_str = strategy_hybrid.get_strat_param(self.strategy_params, 'daily_pnl_risk_adjustment', "-500=0.5,0=1.0,1000=1.5", str, config_parser=self.config, section_config='dynamic_risk')
        self.drm_pnl_adjustment_points = self._parse_daily_pnl_adjustment(daily_pnl_adj_str)
        
        self.min_risk_multiplier = strategy_hybrid.get_strat_param(self.strategy_params, 'min_risk_multiplier', 0.1, float, config_parser=self.config, section_config='dynamic_risk')
        
        self.drm_setup_multipliers = {
            setup.upper(): strategy_hybrid.get_strat_param(
                self.strategy_params, f'setup_size_multiplier_{setup.lower()}',
                self.config.getfloat('dynamic_risk', f'setup_size_multiplier_{setup.lower()}', fallback=default_mult),
                value_type=float
            ) for setup, default_mult in [('A', 1.0), ('B', 0.6), ('C', 0.3), ('NONE', 0.1)]
        }
        
        self.session_risk_multipliers = {}
        sessions_config = ['asia', 'london', 'ny', 'london_ny_overlap', 'off_session']
        defaults_session = {'asia': 0.7, 'london': 1.0, 'ny': 1.2, 'london_ny_overlap': 1.3, 'off_session': 0.5}
        for session in sessions_config:
            key_cfg = f'session_risk_multiplier_{session}'
            self.session_risk_multipliers[session] = strategy_hybrid.get_strat_param(
                self.strategy_params, key_cfg,
                self.config.getfloat('dynamic_risk', key_cfg, fallback=defaults_session.get(session, 1.0)),
                value_type=float
            )


        # Time index determination (de TA_BT)
        self.time_index = None
        min_start_date = pd.Timestamp.max.tz_localize('UTC') # Ensure timezone aware
        valid_pairs = 0
        for pair, df_asset in self.prepared_data.items():
            if not df_asset.empty and isinstance(df_asset.index, pd.DatetimeIndex):
                valid_pairs += 1
                # Ensure index is timezone-aware (UTC)
                if df_asset.index.tzinfo is None or df_asset.index.tzinfo.utcoffset(df_asset.index[0]) is None:
                     self.prepared_data[pair] = df_asset.tz_localize('UTC') # Assume UTC if naive
                elif df_asset.index.tzinfo != timezone.utc:
                     self.prepared_data[pair] = df_asset.tz_convert('UTC') # Convert to UTC

                df_asset = self.prepared_data[pair] # Get the updated (potentially timezone-aware) df

                required_base_cols = [self.open_col, self.high_col, self.low_col, self.close_col, self.volume_col,
                                      self.rh_col, self.ra_col, self.sentiment_col]
                required_ml_cols = self.all_feature_columns # All features are needed for potential scaling/selection
                
                missing_cols = set(required_base_cols + required_ml_cols) - set(df_asset.columns)
                if missing_cols:
                    logger.warning(f"Data for {pair} missing columns: {missing_cols}. This pair may be skipped or cause errors.")
                    # Decide if to continue with this pair or skip. For now, let it try.
                
                min_start_date = min(min_start_date, df_asset.index.min())
                if self.time_index is None:
                    self.time_index = df_asset.index
                else:
                    # Create a union of all timestamps to form the global timeline
                    self.time_index = self.time_index.union(df_asset.index)
        
        if valid_pairs == 0 or self.time_index is None or min_start_date == pd.Timestamp.max.tz_localize('UTC'):
            raise ValueError("prepared_data is empty, has no valid pairs, or could not establish a time index.")

        self.time_index = self.time_index.sort_values() # Sort the global timeline

        # Determine the first valid index to start the loop
        self.start_idx_in_time_index = self.time_index.get_loc(min_start_date, method='bfill')
        
        if self.start_idx_in_time_index == -1: # get_loc can return -1 if not found (though bfill should prevent)
            raise ValueError(f"Min start date {min_start_date} not found in global time index.")

        logger.info(f"Backtester initialized. Global timeline: {len(self.time_index)} steps. "
                    f"Running from {self.time_index[self.start_idx_in_time_index]} to {self.time_index[-1]}.")
        logger.debug(f"Strategy params used for backtest run: {self.strategy_params}")
        logger.debug(f"DRM Setup Multipliers: {self.drm_setup_multipliers}")
        logger.debug(f"DRM PNL Adjustment Points: {self.drm_pnl_adjustment_points}")
        logger.debug(f"DRM Session Multipliers: {self.session_risk_multipliers}")

        # Pre-calculating ATR series for each asset
        atr_period = self.config.getint('strategy_hybrid', 'atr_period_sl_tp', fallback=14)
        self.atr_series_map = {}
        for asset_symbol in self.assets_to_backtest:
            full_asset_data = self.prepared_data.get(asset_symbol)
            if full_asset_data is not None and not full_asset_data.empty and \
               all(col in full_asset_data.columns for col in [self.high_col, self.low_col, self.close_col]):
                self.atr_series_map[asset_symbol] = calculate_atr(
                    full_asset_data[self.high_col],
                    full_asset_data[self.low_col],
                    full_asset_data[self.close_col],
                    time_period=atr_period
                )
            else:
                self.atr_series_map[asset_symbol] = pd.Series(dtype=float)

    # --- Helper methods from B_BT and TA_BT, consolidated ---
    def _parse_daily_pnl_adjustment(self, config_str: str) -> List[Tuple[float, float]]:
        """ Parses 'pnl_threshold=multiplier,pnl_threshold=multiplier' into sorted list of tuples. """
        # (Identique à TA_BT, repris pour clarté)
        try:
            if not config_str or config_str.strip() == '': return [(float('-inf'), 1.0)]
            tiers = []
            for pair in config_str.split(','):
                if '=' not in pair: continue
                threshold_str, multiplier_str = pair.split('=')
                tiers.append((float(threshold_str.strip()), float(multiplier_str.strip())))
            if not tiers: return [(float('-inf'), 1.0)]
            return sorted(tiers, key=lambda x: x[0])
        except Exception as e:
            logger.error(f"Error parsing daily PnL adjustment config: {e}. Using default.")
            return [(float('-inf'), 1.0)]

    def _get_daily_pnl_multiplier(self, today_pnl: float) -> float:
        # (Identique à TA_BT, repris pour clarté)
        multiplier = 1.0
        for threshold, tier_multiplier in self.drm_pnl_adjustment_points:
            if today_pnl >= threshold: multiplier = tier_multiplier
            else: break
        return multiplier

    def _get_session_multiplier(self, current_ts: pd.Timestamp) -> float:
        # (Adapté de B_BT et TA_BT)
        if current_ts.tzinfo is None or current_ts.tzinfo.utcoffset(current_ts) is None:
            current_ts_utc = current_ts.replace(tzinfo=timezone.utc)
        else:
            current_ts_utc = current_ts.astimezone(timezone.utc)
        hour_utc = current_ts_utc.hour

        # Utiliser les définitions d'heures de session de config ou strategy_params
        london_hours_str = strategy_hybrid.get_strat_param(self.strategy_params, 'session_london_hours_utc', self.config.get('dynamic_risk', 'session_london_hours_utc', fallback='7-16'), value_type=str)
        ny_hours_str = strategy_hybrid.get_strat_param(self.strategy_params, 'session_ny_hours_utc', self.config.get('dynamic_risk', 'session_ny_hours_utc', fallback='13-21'), value_type=str)
        asia_hours_str = strategy_hybrid.get_strat_param(self.strategy_params, 'session_asia_hours_utc', self.config.get('dynamic_risk', 'session_asia_hours_utc', fallback='0-8'), value_type=str) # Couvre minuit

        try:
            parse_hours = lambda s: tuple(map(int, s.split('-')))
            london_start, london_end = parse_hours(london_hours_str)
            ny_start, ny_end = parse_hours(ny_hours_str)
            asia_s, asia_e = parse_hours(asia_hours_str) # Renommé pour éviter conflit avec session key
        except ValueError:
            logger.error("Invalid hour format in session config. Using default multiplier (1.0).")
            return 1.0

        is_london = (london_start <= hour_utc < london_end)
        is_ny = (ny_start <= hour_utc < ny_end)
        # Pour l'Asie, gérer le cas où la session chevauche minuit (ex: 22h - 06h)
        if asia_s < asia_e: # Session dans la même journée (ex: 0-8)
            is_asia = (asia_s <= hour_utc < asia_e)
        else: # Session chevauchant minuit (ex: 22-6)
            is_asia = (hour_utc >= asia_s or hour_utc < asia_e)


        session_key_prefix = 'session_risk_multiplier_'
        current_session_name = 'off_session' # Fallback

        if is_london and is_ny: current_session_name = 'london_ny_overlap'
        elif is_london: current_session_name = 'london'
        elif is_ny: current_session_name = 'ny'
        elif is_asia: current_session_name = 'asia'
        
        return self.session_risk_multipliers.get(current_session_name, 1.0)


    def _update_position_exit_levels(self, position: BacktestPosition, current_bar_data: pd.Series) -> BacktestPosition:
        """Updates SL for a position based on Move-to-BE and Trailing Stop logic. (Adapté de B_BT)"""
        if position.exit_price is not None: return position # Already closed

        # current_price = current_bar_data[self.close_col] # Pas utilisé directement, high/low sont plus pertinents pour triggers
        current_high = current_bar_data[self.high_col]
        current_low = current_bar_data[self.low_col]

        if position.side == 'long':
            position.highest_price_since_entry = max(position.highest_price_since_entry, current_high)
            
            # Move to BE
            if not position.moved_to_be and self.move_to_be_profit_percent > 0:
                profit_target_for_be = position.entry_price * (1 + self.move_to_be_profit_percent)
                if current_high >= profit_target_for_be:
                    # BE price could include estimated fees for two trades / qty
                    be_price_estimate = position.entry_price * (1 + 2 * self.fee_percent * self.slippage_percent) # Approximation simple
                    if position.sl_price is None or be_price_estimate > position.sl_price:
                        logger.debug(f"BACKTEST [{position.symbol_market} @ {current_bar_data.name}]: Move-to-BE. Old SL: {position.sl_price}, New SL: {be_price_estimate:.4f}")
                        position.sl_price = be_price_estimate
                        position.moved_to_be = True
                        position.trailing_sl_active = False # TS désactivé si BE est atteint
            
            # Trailing Stop (seulement si pas en BE)
            if not position.moved_to_be and self.trailing_stop_profit_percent_start > 0 and self.trailing_stop_distance_percent > 0:
                if not position.trailing_sl_active: # Activation du TSL
                    profit_target_for_tsl_start = position.entry_price * (1 + self.trailing_stop_profit_percent_start)
                    if position.highest_price_since_entry >= profit_target_for_tsl_start:
                        position.trailing_sl_active = True
                        logger.debug(f"BACKTEST [{position.symbol_market} @ {current_bar_data.name}]: Trailing SL ACTIVATED. Highest: {position.highest_price_since_entry:.4f}")
                
                if position.trailing_sl_active:
                    new_trailing_sl = position.highest_price_since_entry * (1 - self.trailing_stop_distance_percent)
                    if position.sl_price is None or new_trailing_sl > position.sl_price:
                        logger.debug(f"BACKTEST [{position.symbol_market} @ {current_bar_data.name}]: Trailing SL UPDATED. Highest: {position.highest_price_since_entry:.4f}, New SL: {new_trailing_sl:.4f}")
                        position.sl_price = new_trailing_sl
                        position.trailing_sl_price = new_trailing_sl
        
        # TODO: Implémenter la logique pour les positions SHORT de manière symétrique
        elif position.side == 'short':
            position.lowest_price_since_entry = min(position.lowest_price_since_entry, current_low)
            # Move to BE for short
            if not position.moved_to_be and self.move_to_be_profit_percent > 0:
                profit_target_for_be = position.entry_price * (1 - self.move_to_be_profit_percent)
                if current_low <= profit_target_for_be:
                    be_price_estimate = position.entry_price * (1 - 2 * self.fee_percent * self.slippage_percent)
                    if position.sl_price is None or be_price_estimate < position.sl_price:
                        position.sl_price = be_price_estimate
                        position.moved_to_be = True
                        position.trailing_sl_active = False
            
            # Trailing Stop for short
            if not position.moved_to_be and self.trailing_stop_profit_percent_start > 0 and self.trailing_stop_distance_percent > 0:
                if not position.trailing_sl_active:
                    profit_target_for_tsl_start = position.entry_price * (1 - self.trailing_stop_profit_percent_start)
                    if position.lowest_price_since_entry <= profit_target_for_tsl_start:
                        position.trailing_sl_active = True
                
                if position.trailing_sl_active:
                    new_trailing_sl = position.lowest_price_since_entry * (1 + self.trailing_stop_distance_percent)
                    if position.sl_price is None or new_trailing_sl < position.sl_price:
                        position.sl_price = new_trailing_sl
                        position.trailing_sl_price = new_trailing_sl
        return position

    def _get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculates total portfolio value (cash + open positions). (De B_BT)"""
        value = self.cash
        for pos in self.open_positions:
            current_price = current_prices.get(pos.symbol_market, pos.entry_price)
            if pos.side == 'long': value += pos.qty * current_price
            elif pos.side == 'short': value += pos.qty * pos.entry_price + (pos.qty * (pos.entry_price - current_price))
        return value

    def _open_position(self, symbol_market: str, side: str, qty: float, entry_price_simulated: float, 
                       tp_price: Optional[float], sl_price: Optional[float],
                       timestamp: pd.Timestamp,
                       setup_quality: str, initial_risk_multiplier: float) -> bool:
        """Simulates opening a new position. (Adapté de B_BT)"""
        if qty <= 0 or entry_price_simulated <= 0: return False

        cost = qty * entry_price_simulated
        fee_on_open = cost * self.fee_percent

        if side == 'long':
            if self.cash < cost + fee_on_open:
                logger.debug(f"BACKTEST [{symbol_market} @ {timestamp}]: Insufficient cash ({self.cash:.2f}) for LONG. Cost+Fee: {cost+fee_on_open:.2f}")
                return False
            self.cash -= (cost + fee_on_open)
        elif side == 'short': # Simplifié, pas de gestion de marge pour l'instant
            if self.cash < fee_on_open:
                logger.debug(f"BACKTEST [{symbol_market} @ {timestamp}]: Insufficient cash ({self.cash:.2f}) for SHORT fee. Fee: {fee_on_open:.2f}")
                return False
            self.cash -= fee_on_open # Déduire les frais, la valeur de la position sera ajoutée à la clôture

        position = BacktestPosition(
            symbol_market=symbol_market, entry_price=entry_price_simulated, qty=qty, side=side,
            tp_price=tp_price, sl_price=sl_price, entry_timestamp=timestamp,
            setup_quality=setup_quality, initial_risk_multiplier=initial_risk_multiplier,
            fee=fee_on_open # Fee initiale
        )
        self.open_positions.append(position)
        logger.debug(f"BACKTEST [{symbol_market} @ {timestamp}]: OPEN {side.upper()} Qty:{qty:.4f} @{entry_price_simulated:.4f} TP:{tp_price} SL:{sl_price}. Fee:{fee_on_open:.4f}. Setup:{setup_quality}, RiskM:{initial_risk_multiplier:.2f}")
        return True

    def _close_position(self, position_idx: int, exit_price_simulated: float, timestamp: pd.Timestamp, reason: str) -> Optional[BacktestPosition]:
        """Simulates closing an open position. (Adapté de B_BT)"""
        if not (0 <= position_idx < len(self.open_positions)): return None

        pos = self.open_positions.pop(position_idx)
        
        proceeds = pos.qty * exit_price_simulated
        fee_on_close = proceeds * self.fee_percent
        pos.fee += fee_on_close # Cumuler les frais

        if pos.side == 'long':
            self.cash += (proceeds - fee_on_close) # cash in - fee
            pos.pnl = (exit_price_simulated - pos.entry_price) * pos.qty - pos.fee
        elif pos.side == 'short':
            # Pour short: Cash In = Qty * EntryPrice (reçu à l'ouverture, hors frais)
            # Cash Out = Qty * ExitPrice (payé à la clôture)
            # Donc PnL = (Entry - Exit) * Qty - Fees
            # Changement de cash = (Qty*Entry - FraisOuverture) + ( (Qty*Entry) - (Qty*Exit + FraisCloture) )
            # Cash initial pour short: on ne déduit que les frais. Le "produit de la vente" n'est pas ajouté au cash directement.
            # À la clôture, on règle: cash += (pos.qty * pos.entry_price) - (proceeds + fee_on_close)
            # Ou plus simple: le PnL est ajouté au cash.
            pnl_before_fees_adj = (pos.entry_price - exit_price_simulated) * pos.qty
            pos.pnl = pnl_before_fees_adj - pos.fee
            self.cash += (pos.qty * pos.entry_price) # Retour du "collatéral" de la vente à découvert
            self.cash += (pnl_before_fees_adj - fee_on_close) # Ajustement du cash avec PnL net de frais de clôture. Frais d'ouverture déjà déduits.


        pos.pnl_percent = (pos.pnl / (pos.entry_price * pos.qty)) * 100 if (pos.entry_price * pos.qty) != 0 else 0.0
        pos.exit_price = exit_price_simulated
        pos.exit_timestamp = timestamp
        pos.exit_reason = reason
        
        self.closed_trades.append(pos)
        logger.debug(f"BACKTEST [{pos.symbol_market} @ {timestamp}]: CLOSE {pos.side.upper()} @{exit_price_simulated:.4f} ({reason}). PnL:{pos.pnl:.2f} ({pos.pnl_percent:.2f}%). TotalFee:{pos.fee:.4f}")
        return pos

    def _log_daily_snapshot(self, timestamp: pd.Timestamp, current_prices: Dict[str, float]):
        """Logs daily portfolio value. (De B_BT)"""
        current_value = self._get_portfolio_value(current_prices)
        daily_pnl = 0.0
        if self.daily_portfolio_history: # Si l'historique n'est pas vide
            # PnL par rapport à la valeur du snapshot précédent
            daily_pnl = current_value - self.daily_portfolio_history[-1]['portfolio_value']
        
        self.daily_portfolio_history.append({
            'timestamp': timestamp,
            'portfolio_value': current_value,
            'cash': self.cash,
            'open_positions_count': len(self.open_positions),
            'daily_pnl': daily_pnl
        })

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculates final performance metrics. (De B_BT, légèrement adapté)"""
        if not self.daily_portfolio_history:
            logger.warning("Portfolio history empty. Cannot calculate metrics.")
            # Retourner un dict avec des NaNs ou zéros pour la structure
            return {
                "initial_capital": self.initial_capital, "final_portfolio_value": self.cash,
                "total_return_percent": 0.0, "sharpe_ratio": np.nan, "max_drawdown_percent": np.nan,
                "total_net_pnl": 0.0, "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "win_rate_percent": 0.0, "average_pnl_per_trade": 0.0, "average_winning_trade": 0.0,
                "average_losing_trade": 0.0, "profit_factor": np.nan, "total_fees_paid": 0.0
            }

        history_df = pd.DataFrame(self.daily_portfolio_history)
        if 'timestamp' in history_df.columns:
            history_df = history_df.set_index('timestamp')
        else: # Devrait toujours y avoir un timestamp
            logger.error("Timestamp manquant dans daily_portfolio_history.")
            return {}


        final_value = history_df['portfolio_value'].iloc[-1] if not history_df.empty else self.initial_capital
        total_return_pct = ((final_value / self.initial_capital) - 1) * 100 if self.initial_capital > 0 else 0.0
        
        returns = history_df['portfolio_value'].pct_change().dropna()
        sharpe_ratio = np.nan
        max_drawdown_pct = np.nan

        if not returns.empty:
            trading_days_in_year = 252 # Standard assumption
            # More robust: calculate actual trading days from index if history_df has daily frequency
            if len(history_df.index.unique()) > 1:
                time_span_days_approx = (history_df.index.max() - history_df.index.min()).days
                if time_span_days_approx > 0: # Avoid division by zero
                    # This annualization factor works if returns are daily. If not, it's an approximation.
                    annualization_factor = trading_days_in_year # For daily returns
                    # If returns are, e.g., hourly, this needs adjustment.
                    # Example: if 24 data points per day, then sqrt(trading_days_in_year * 24)
                    # For simplicity, assuming daily or near-daily snapshots in daily_portfolio_history
            else: # Not enough data for annualization
                annualization_factor = 1


            mean_return = returns.mean()
            std_return = returns.std()

            if std_return != 0 and not np.isnan(std_return):
                sharpe_ratio = (mean_return / std_return) * np.sqrt(annualization_factor) # Assuming 0 risk-free rate

            # Max Drawdown
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns / peak) - 1
            max_drawdown_pct = drawdown.min() * 100 if not drawdown.empty else 0.0

        num_trades = len(self.closed_trades)
        total_fees = sum(t.fee for t in self.closed_trades)
        
        winning_trades = sum(1 for t in self.closed_trades if t.pnl is not None and t.pnl > 0)
        losing_trades = sum(1 for t in self.closed_trades if t.pnl is not None and t.pnl < 0)
        win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0.0
        
        total_pnl = sum(t.pnl for t in self.closed_trades if t.pnl is not None)
        avg_pnl_per_trade = total_pnl / num_trades if num_trades > 0 else 0.0
        avg_win_trade = sum(t.pnl for t in self.closed_trades if t.pnl is not None and t.pnl > 0) / winning_trades if winning_trades > 0 else 0.0
        avg_loss_trade = sum(t.pnl for t in self.closed_trades if t.pnl is not None and t.pnl < 0) / losing_trades if losing_trades > 0 else 0.0 # Will be negative
        
        sum_profits = sum(t.pnl for t in self.closed_trades if t.pnl is not None and t.pnl > 0)
        sum_losses_abs = abs(sum(t.pnl for t in self.closed_trades if t.pnl is not None and t.pnl < 0))
        profit_factor = sum_profits / sum_losses_abs if sum_losses_abs > 0 else (np.inf if sum_profits > 0 else np.nan)


        metrics = {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": final_value,
            "total_return_percent": total_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_percent": max_drawdown_pct,
            "total_net_pnl": total_pnl,
            "total_trades": num_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_percent": win_rate,
            "average_pnl_per_trade": avg_pnl_per_trade,
            "average_winning_trade": avg_win_trade,
            "average_losing_trade": avg_loss_trade,
            "profit_factor": profit_factor,
            "total_fees_paid": total_fees
        }
        logger.info("--- Backtest Performance Metrics ---")
        for k, v_val in metrics.items(): # Renamed v to v_val to avoid conflict
            if isinstance(v_val, float): logger.info(f"  {k.replace('_', ' ').title()}: {v_val:.4f}")
            else: logger.info(f"  {k.replace('_', ' ').title()}: {v_val}")
        logger.info("------------------------------------")
        return metrics

    def run_backtest(self) -> Dict:
        """Executes the backtest simulation loop. (Principalement de TA_BT, modifié)"""
        logger.info("="*40)
        logger.info(" DÉBUT DU BACKTEST (Version Fusionnée) ")
        logger.info(f"Capital initial: {self.initial_capital:.2f}")
        logger.info(f"Frais de transaction: {self.fee_percent*100:.3f}% (par transaction)")
        logger.info(f"Slippage simulé: {self.slippage_percent*100:.3f}% (par transaction)")

        current_day_ts_obj = None # Pour le suivi PnL journalier (Utiliser Timestamp, pas str)
        today_pnl_for_drm = 0.0

        # Log initial portfolio state (avant la première barre)
        initial_prices_snapshot = {}
        if not self.time_index.empty:
            first_global_ts_for_snapshot = self.time_index[self.start_idx_in_time_index]
            for asset_s, asset_df_s in self.prepared_data.items():
                if first_global_ts_for_snapshot in asset_df_s.index:
                    initial_prices_snapshot[asset_s] = asset_df_s.loc[first_global_ts_for_snapshot, self.close_col]
                else: # Si l'actif n'a pas de données à ce timestamp exact, essayer ffill ou prendre le premier dispo
                    available_ts = asset_df_s.index[asset_df_s.index <= first_global_ts_for_snapshot]
                    if not available_ts.empty:
                        initial_prices_snapshot[asset_s] = asset_df_s.loc[available_ts.max(), self.close_col]
                    else: initial_prices_snapshot[asset_s] = 0 # Fallback
            self._log_daily_snapshot(first_global_ts_for_snapshot - pd.Timedelta(seconds=1), initial_prices_snapshot)


        # Loop through time steps using the global timeline
        # TA_BT's loop was `range(start_idx, len(self.time_index) - 1)` to use `current_ts` and `next_ts`.
        # B_BT's loop was simpler. Let's adapt TA_BT's structure slightly.
        # We need a "next bar" concept for execution simulation.
        # The loop will go up to len-2 to allow fetching current_ts and next_ts within the loop.
        # If we need to process the very last bar, special handling might be needed after the loop for `_log_daily_snapshot`.
        
        for i in tqdm(range(self.start_idx_in_time_index, len(self.time_index)), desc="Running Backtest"):
            current_ts = self.time_index[i] # Decision point time 't'
            # `next_ts` is used to simulate execution within the *next* bar.
            # So, decisions at `current_ts` are executed based on OHLC of `current_ts` (if intra-bar)
            # or Open of `next_ts`.
            # Let's use `current_ts` for decisions and simulate execution using `current_ts`'s OHLC data.
            # This means SL/TP can be hit intra-bar. Entries can be at Open/Close of `current_ts`.
            # This simplifies not needing `next_ts` explicitly in the loop for fetching data,
            # as `current_bar_data` will contain the OHLC of the current period.

            # --- Daily PnL Reset for DRM ---
            if current_day_ts_obj is None or current_ts.date() != current_day_ts_obj.date():
                if current_day_ts_obj is not None: # Fin de la journée précédente
                    # Snapshotting is now done per bar, see below
                    pass
                current_day_ts_obj = current_ts 
                today_pnl_for_drm = 0.0


            # --- Process each asset for the current_ts ---
            asset_decisions_this_bar: Dict[str, Dict] = {} # Store decisions for this bar
            current_prices_this_bar: Dict[str, float] = {} # Store close prices for portfolio valuation

            for asset_symbol in self.assets_to_backtest:
                if asset_symbol not in self.prepared_data or current_ts not in self.prepared_data[asset_symbol].index:
                    # logger.debug(f"No data for {asset_symbol} at {current_ts}. Skipping.")
                    continue

                asset_df = self.prepared_data[asset_symbol]
                current_bar_data = asset_df.loc[current_ts]
                current_price = current_bar_data[self.close_col]
                current_prices_this_bar[asset_symbol] = current_price

                # --- Reconstruct MS/OF features for strategy_hybrid (de B_BT) ---
                # Assuming MS/OF features are already columns in current_bar_data (from training_pipeline)
                ms_features = {k: v for k, v in current_bar_data.items() if k.startswith('ms_')}
                of_features = {k: v for k, v in current_bar_data.items() if k.startswith('of_')}

                # --- ML Model Predictions (inspiré de B_BT & TA_BT) ---
                xgb_prediction_probs = None
                if self.ml_ready and self.xgb_model and self.xgb_feature_names and self.feature_scaler:
                    # feature_engineer.build_feature_vector_for_xgboost expects pd.Series, List[str], Scaler
                    # current_bar_data already contains all raw features combined
                    xgb_input_df = feature_engineer.build_feature_vector_for_xgboost(
                        current_bar_data, self.xgb_feature_names, self.feature_scaler
                    )
                    if xgb_input_df is not None and not xgb_input_df.isnull().values.any().any():
                        # Convert to DMatrix for XGBoost native API
                        dmatrix_live = xgboost.DMatrix(xgb_input_df, feature_names=self.xgb_feature_names)
                        raw_xgb_preds = self.xgb_model.predict(dmatrix_live)
                        xgb_prediction_probs = raw_xgb_preds[0] if raw_xgb_preds.ndim > 1 and raw_xgb_preds.shape[0] > 0 else raw_xgb_preds
                    # else: logger.debug(f"XGB input generation failed or has NaNs for {asset_symbol} at {current_ts}")
                
                fq_predicted_ratios = None
                if self.ml_ready and self.fq_model and self.fq_feature_names and self.feature_scaler and self.fq_quantiles:
                    fq_config_params = self.config['futurequant'] if 'futurequant' in self.config.sections() else {}
                    # build_feature_sequence_for_fq expects historical DataFrame, config, scaler, feature_names, current_ts
                    fq_input_sequence = feature_engineer.build_feature_sequence_for_fq(
                        asset_df, fq_config_params, self.feature_scaler, self.fq_feature_names, current_ts
                    )
                    if fq_input_sequence is not None:
                        raw_fq_preds = self.fq_model.predict(fq_input_sequence)
                        fq_predicted_ratios = raw_fq_preds[0] if raw_fq_preds.ndim > 1 and raw_fq_preds.shape[0] > 0 else raw_fq_preds
                    # else: logger.debug(f"FQ input generation failed for {asset_symbol} at {current_ts}")


                # --- Get current position state for this asset ---
                current_pos_for_asset_qty = 0.0
                open_pos_idx_for_asset = -1
                for idx, p_obj in enumerate(self.open_positions):
                    if p_obj.symbol_market == asset_symbol:
                        current_pos_for_asset_qty = p_obj.qty if p_obj.side == 'long' else -p_obj.qty
                        open_pos_idx_for_asset = idx
                        break
                
                # --- Retrieve ATR value for current_ts and asset_symbol ---
                atr_value_for_decision = None
                asset_specific_atr_series = self.atr_series_map.get(asset_symbol)
                if asset_specific_atr_series is not None and not asset_specific_atr_series.empty:
                    atr_value_for_decision = asset_specific_atr_series.get(current_ts)
                    if pd.isna(atr_value_for_decision):
                        # Fallback to the last valid ATR at or before current_ts
                        relevant_atrs_up_to_current_ts = asset_specific_atr_series.loc[:current_ts].dropna()
                        if not relevant_atrs_up_to_current_ts.empty:
                            atr_value_for_decision = relevant_atrs_up_to_current_ts.iloc[-1]
                            logger.debug(f"[{asset_symbol}] ATR for {current_ts} is NaN. Using last valid ATR: {atr_value_for_decision:.4f}")
                        else:
                            logger.warning(f"[{asset_symbol}] ATR for {current_ts} is NaN and no prior valid ATR found.")
                            atr_value_for_decision = None # Explicitly set to None
                else:
                    logger.warning(f"[{asset_symbol}] No pre-calculated ATR series found or series is empty.")

                # --- Generate Trade Decision (using strategy_hybrid) ---
                # strategy_hybrid needs: symbol_market, current_ratings (from bar_data), latest_price, sentiment_score (from bar_data),
                # xgb_probs, fq_ratios, ms_features, of_features, current_position_qty, wallet, today_pnl, config
                wallet_value_for_decision = self._get_portfolio_value(current_prices_this_bar) # Utiliser prix actuels
                ratings_data_series = current_bar_data # strategy_hybrid va extraire R_H, R_A
                sentiment_val = current_bar_data.get(self.sentiment_col, 0.0)

                decision = strategy_hybrid.generate_trade_decision(
                    symbol_market=asset_symbol, current_ratings=ratings_data_series, latest_price=current_price,
                    sentiment_score=sentiment_val, xgb_probabilities=xgb_prediction_probs,
                    fq_predicted_ratios=fq_predicted_ratios, market_structure_features=ms_features,
                    order_flow_features=of_features, atr_value=atr_value_for_decision, # <<< Added atr_value
                    current_position_qty=current_pos_for_asset_qty,
                    wallet=wallet_value_for_decision, today_pnl=today_pnl_for_drm, config=self.config
                )
                asset_decisions_this_bar[asset_symbol] = decision

            # --- Process Open Positions (SL/TP hits, Exit Level Updates) for ALL assets ---
            # Iterate backwards to allow safe removal from self.open_positions
            position_closed_by_sl_tp_this_bar_map = {asset: False for asset in self.assets_to_backtest}

            for pos_idx in range(len(self.open_positions) - 1, -1, -1):
                pos = self.open_positions[pos_idx]
                asset_sym = pos.symbol_market
                
                if asset_sym not in self.prepared_data or current_ts not in self.prepared_data[asset_sym].index:
                    continue # No current bar data for this position's asset
                
                bar_data_for_pos = self.prepared_data[asset_sym].loc[current_ts]

                # 1. Update exit levels (Move to BE, Trailing SL) using current_bar_data (OHLC of current_ts)
                # This happens based on the state *before* checking for SL/TP hits in this bar.
                pos = self._update_position_exit_levels(pos, bar_data_for_pos)
                # self.open_positions[pos_idx] = pos # Dataclass is mutable, changes reflect.

                # 2. Check for SL/TP hit within the current bar (current_ts)
                exit_reason = None; actual_exit_price = None
                bar_low = bar_data_for_pos[self.low_col]
                bar_high = bar_data_for_pos[self.high_col]

                if pos.side == 'long':
                    if pos.sl_price is not None and bar_low <= pos.sl_price:
                        exit_reason, actual_exit_price = 'SL', min(pos.sl_price, bar_high) # Exit at SL, or bar high if SL gapped through
                    elif pos.tp_price is not None and bar_high >= pos.tp_price:
                        exit_reason, actual_exit_price = 'TP', max(pos.tp_price, bar_low) # Exit at TP, or bar low if TP gapped through
                elif pos.side == 'short':
                    if pos.sl_price is not None and bar_high >= pos.sl_price:
                        exit_reason, actual_exit_price = 'SL', max(pos.sl_price, bar_low)
                    elif pos.tp_price is not None and bar_low <= pos.tp_price:
                        exit_reason, actual_exit_price = 'TP', min(pos.tp_price, bar_high)
                
                if exit_reason and actual_exit_price is not None:
                    # Apply slippage to actual_exit_price
                    simulated_exit_price_after_slippage = actual_exit_price * (1 - self.slippage_percent) if pos.side == 'long' else actual_exit_price * (1 + self.slippage_percent)
                    
                    closed_pos_obj = self._close_position(pos_idx, simulated_exit_price_after_slippage, current_ts, exit_reason)
                    if closed_pos_obj and closed_pos_obj.pnl is not None: today_pnl_for_drm += closed_pos_obj.pnl
                    position_closed_by_sl_tp_this_bar_map[asset_sym] = True


            # --- Process Decisions (Open new, Close existing by signal) for ALL assets ---
            open_positions_count_current = len(self.open_positions) # Re-evaluate after SL/TP closures

            for asset_symbol, decision_for_asset in asset_decisions_this_bar.items():
                if asset_symbol not in self.prepared_data or current_ts not in self.prepared_data[asset_symbol].index:
                    continue
                
                bar_data_for_decision_asset = self.prepared_data[asset_symbol].loc[current_ts]
                price_at_decision_open = bar_data_for_decision_asset[self.open_col] # Use Open for entries/signal exits
                price_at_decision_close = bar_data_for_decision_asset[self.close_col]


                action = decision_for_asset.get('action', 'HOLD')
                setup_quality = decision_for_asset.get('setup_quality', 'NONE')

                # Check if position was already closed by SL/TP in this same bar
                if position_closed_by_sl_tp_this_bar_map.get(asset_symbol, False):
                    logger.debug(f"Position for {asset_symbol} was closed by SL/TP in this bar. Skipping further action based on signal.")
                    continue
                
                # Find if there's an existing open position for this asset
                existing_pos_idx = -1
                for idx_check, p_obj_check in enumerate(self.open_positions):
                    if p_obj_check.symbol_market == asset_symbol:
                        existing_pos_idx = idx_check
                        break
                
                if action == 'BUY' and existing_pos_idx == -1: # Open new long
                    if open_positions_count_current < self.max_concurrent_positions:
                        # --- Dynamic Position Sizing ---
                        base_size_mult = self.drm_setup_multipliers.get(setup_quality.upper(), self.drm_setup_multipliers['NONE'])
                        session_mult = self._get_session_multiplier(current_ts)
                        pnl_mult = self._get_daily_pnl_multiplier(today_pnl_for_drm)
                        final_risk_multiplier = base_size_mult * session_mult * pnl_mult
                        final_risk_multiplier = max(self.min_risk_multiplier, final_risk_multiplier)
                        
                        pos_size_conf_percent = strategy_hybrid.get_strat_param(self.strategy_params, 'position_size_percent', self.config.getfloat('trading', 'position_size_percent', fallback=0.02), value_type=float)
                        
                        current_total_portfolio_value = self._get_portfolio_value(current_prices_this_bar) # Value at *this* bar's close
                        target_trade_value = current_total_portfolio_value * pos_size_conf_percent * final_risk_multiplier
                        
                        # Simulate entry at Open of current_ts bar + slippage
                        simulated_entry_price_with_slippage = price_at_decision_open * (1 + self.slippage_percent)

                        qty_to_open = target_trade_value / simulated_entry_price_with_slippage if simulated_entry_price_with_slippage > 0 else 0
                        
                        if qty_to_open > 0:
                            # Recalculate TP/SL based on simulated_entry_price_with_slippage
                            # The decision dictionary contains TP/SL based on `latest_price` (which was current_bar_data[self.close_col] at decision time).
                            # It's better to recalculate TP/SL based on the actual simulated entry price.
                            sl_dist_pct_param = strategy_hybrid.get_strat_param(self.strategy_params, 'sl_distance_percent', 0.015, float, config_parser=self.config, section_config='strategy_hybrid')
                            tp_sl_ratio_param = strategy_hybrid.get_strat_param(self.strategy_params, 'tp_sl_ratio', indicators.PHI, float, config_parser=self.config, section_config='strategy_hybrid')

                            recalc_tp_price, recalc_sl_price = calculate_fib_tp_sl_prices(
                                entry_price=simulated_entry_price_with_slippage,
                                is_long=True, # Assuming BUY is long
                                sl_distance_percent=sl_dist_pct_param,
                                tp_sl_ratio=tp_sl_ratio_param
                            )

                            if self._open_position(
                                asset_symbol, 'long', qty_to_open, simulated_entry_price_with_slippage,
                                recalc_tp_price, recalc_sl_price, # decision_for_asset.get('tp_price'), decision_for_asset.get('sl_price'),
                                current_ts, 
                                setup_quality, final_risk_multiplier
                            ):
                                open_positions_count_current +=1
                    else:
                        logger.debug(f"Max concurrent positions ({self.max_concurrent_positions}) reached. Cannot open new position for {asset_symbol}.")

                elif action == 'SELL_TO_CLOSE_LONG' and existing_pos_idx != -1:
                    # Simulate exit at Open of current_ts bar - slippage
                    simulated_exit_price_with_slippage = price_at_decision_open * (1 - self.slippage_percent)
                    closed_pos_obj = self._close_position(existing_pos_idx, simulated_exit_price_with_slippage, current_ts, 'Signal')
                    if closed_pos_obj and closed_pos_obj.pnl is not None: today_pnl_for_drm += closed_pos_obj.pnl
                    # open_positions_count_current will be updated naturally in next iter or if loop ends

                # TODO: Add logic for SELL (short entry) and BUY_TO_CLOSE_SHORT

            # --- End of bar processing for all assets ---
            # Log portfolio snapshot using close prices of the current bar
            self._log_daily_snapshot(current_ts, current_prices_this_bar)


        # --- End of backtest loop (all timestamps processed) ---
        logger.info(f"\nFin de la simulation. Clôture de {len(self.open_positions)} positions ouvertes restantes...")
        if self.time_index.empty: # Should not happen if loop ran
            logger.warning("Time index vide à la fin du backtest. Aucune donnée traitée.")
        elif len(self.open_positions) > 0 :
            last_processed_ts = self.time_index[len(self.time_index)-1] # The actual last timestamp processed in the loop
            
            # Utiliser les prix de clôture du dernier timestamp traité pour fermer les positions
            last_prices_for_closure = {}
            for asset_sym_close, asset_df_close in self.prepared_data.items():
                if last_processed_ts in asset_df_close.index:
                    last_prices_for_closure[asset_sym_close] = asset_df_close.loc[last_processed_ts, self.close_col]
                else: # Fallback if an asset stopped having data before others
                    available_ts_close = asset_df_close.index[asset_df_close.index <= last_processed_ts]
                    if not available_ts_close.empty:
                         last_prices_for_closure[asset_sym_close] = asset_df_close.loc[available_ts_close.max(), self.close_col]
                    else: last_prices_for_closure[asset_sym_close] = 0


            for pos_idx in range(len(self.open_positions) -1, -1, -1):
                pos_to_close_eod = self.open_positions[pos_idx]
                exit_price_eod = last_prices_for_closure.get(pos_to_close_eod.symbol_market, pos_to_close_eod.entry_price) # Fallback to entry
                
                # Apply slippage
                simulated_exit_price_eod_slippage = exit_price_eod * (1 - self.slippage_percent) if pos_to_close_eod.side == 'long' else exit_price_eod * (1 + self.slippage_percent)

                closed_pos_final = self._close_position(pos_idx, simulated_exit_price_eod_slippage, last_processed_ts, 'EndOfData')
                if closed_pos_final and closed_pos_final.pnl is not None: today_pnl_for_drm += closed_pos_final.pnl # Affects PnL du dernier jour

        # Final portfolio snapshot after closing all positions
        if not self.time_index.empty:
            last_processed_ts_final_snap = self.time_index[len(self.time_index)-1]
            # Portfolio value should reflect cash after all closures. current_prices don't matter if no open positions.
            self._log_daily_snapshot(last_processed_ts_final_snap, {}) 


        # --- Finalisation et Métriques ---
        final_metrics = self._calculate_performance_metrics()

        # Préparer les DataFrames pour le retour (trades et historique journalier)
        trades_df_final = pd.DataFrame([vars(t) for t in self.closed_trades])
        if not trades_df_final.empty and 'entry_timestamp' in trades_df_final.columns:
            trades_df_final['entry_timestamp'] = pd.to_datetime(trades_df_final['entry_timestamp'])
        if not trades_df_final.empty and 'exit_timestamp' in trades_df_final.columns:
            trades_df_final['exit_timestamp'] = pd.to_datetime(trades_df_final['exit_timestamp'])


        days_df_final = pd.DataFrame(self.daily_portfolio_history)
        if not days_df_final.empty and 'timestamp' in days_df_final.columns:
             days_df_final['timestamp'] = pd.to_datetime(days_df_final['timestamp'])
             # Assurer l'unicité de l'index si on le définit, et qu'il est trié
             days_df_final = days_df_final.drop_duplicates(subset=['timestamp'], keep='last').set_index('timestamp').sort_index()
        
        final_portfolio_value_from_history = self.daily_portfolio_history[-1]['portfolio_value'] if self.daily_portfolio_history else self.initial_capital


        logger.info("="*40)
        logger.info(" FIN DU BACKTEST (Version Fusionnée) ")
        logger.info(f"Final Portfolio Value: {final_portfolio_value_from_history:.2f}")
        logger.info(f"Total Trades: {final_metrics.get('total_trades', 0)}")
        logger.info(f"Total Return: {final_metrics.get('total_return_percent', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {final_metrics.get('sharpe_ratio', np.nan):.4f}")
        logger.info(f"Max Drawdown: {final_metrics.get('max_drawdown_percent', 0):.2f}%")
        logger.info(f"Profit Factor: {final_metrics.get('profit_factor', np.nan):.4f}")
        logger.info("="*40)

        return {
            'metrics': final_metrics,
            'trades': trades_df_final,
            'days': days_df_final, # Daily (or per-bar) snapshots
            'final_cash': self.cash,
            'final_portfolio_value': final_portfolio_value_from_history
        }


# ... (Le reste de trading_analyzer.py : RobustnessAnalyzer, fonctions publiques, plotting)
# La fonction calculate_performance_metrics originale dans trading_analyzer.py peut être supprimée car remplacée par la méthode de classe.

# Modifications à apporter dans les fonctions publiques de trading_analyzer.py:
# - execute_backtest utilisera cette nouvelle classe Backtester.
# - Les fonctions de plotting (plot_equity_vs_asset, plot_backtest_monthly_returns) devront s'adapter au format de sortie
#   (ex: 'days' DataFrame venant de backtest_result['days']).

# Dans execute_backtest:
# backtester_instance = Backtester(...)
# backtest_result = backtester_instance.run_backtest()
# return backtest_result  <-- Ceci est déjà le cas.

# Dans plot_equity_vs_asset:
# days_df = backtest_result['days'].copy()
# equity_series = days_df['portfolio_value'] # Utiliser 'portfolio_value'
# Le reste devrait fonctionner.

# Dans plot_backtest_monthly_returns:
# days_df = backtest_result['days'].copy()
# daily_returns = days_df['portfolio_value'].pct_change().fillna(0)
# monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
# Le reste devrait fonctionner.