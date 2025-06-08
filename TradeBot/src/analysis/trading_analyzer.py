# trading_analyzer.py
import pandas as pd
import numpy as np
import logging
import time
import sys # For stdout in setup_logging
from datetime import datetime, timedelta, timezone
from configparser import ConfigParser, NoSectionError, NoOptionError
from typing import Dict, Optional, List, Tuple
import os
import pickle # For loading scalers
import json # For loading training params
import traceback
from ..indicators import calculate_atr

# --- Imports for Analysis and Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm # Use tqdm.auto for notebook compatibility
import random # For parameter perturbation

# --- Import necessary modules from your project ---
# We need these modules because the Backtester and Strategy_Hybrid will call their functions
import indicators # Module unifié remplaçant pi_ratings
from pipeline import feature_engineer # For feature building
from models import model_xgboost # For XGBoost predictions
from models import model_futurequant # For FutureQuant predictions
from strategies import strategy_hybrid # For strategy decisions
from analysis import market_structure_analyzer # For market structure analysis
from analysis import order_flow_analyzer # For order flow analysis
from pipeline import training_pipeline # For loading training data and models

# --- Configuration and Logging ---
# Config file is global for the module, but functions that need config should receive it.
CONFIG_FILE = 'config.ini'
# Use a specific logger name for this module
logger = logging.getLogger(__name__)

# Global placeholder for prepared_data, used by plotting functions
_prepared_data_global = None

# Global flag for graceful shutdown (less relevant in notebook cells, but good practice)
# shutting_down = False

# --- Helper Functions (Moved from other files or created) ---

def setup_logging(log_level=logging.INFO):
    """Configure basic logging for the analyzer module, suitable for notebooks."""
    # Clear existing handlers to avoid duplicate output in notebooks
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.DEBUG) # Capture all levels internally

    # Console handler with level control
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level) # Set the desired output level
    root_logger.addHandler(console_handler)

    # Optional: File handler can be added here if needed

    logger.info(f"Logging configured for trading_analyzer with console level {logging.getLevelName(log_level)}.")


def get_config_option(config: ConfigParser, section: str, option: str, fallback=None):
    """Safely get a config option."""
    try:
        return config.get(section, option)
    except (NoSectionError, NoOptionError):
        # logger.debug(f"Config option '{option}' in section '{section}' not found. Using fallback: {fallback}") # Too verbose
        if fallback is not None:
            return fallback
        else:
             logger.warning(f"Config option '{option}' in section '{section}' not found and no fallback provided.")
             raise # Re-raise if no fallback, means parameter is essential


def get_config_list_lower(config: ConfigParser, section: str, option: str, fallback: List = []) -> List[str]:
    """Safely get a comma-separated list config option and convert items to lower case."""
    value = get_config_option(config, section, option, fallback=', '.join(fallback))
    if isinstance(value, str):
         return [item.strip().lower() for item in value.split(',') if item.strip()]
    # If fallback wasn't a string, assume it was already a list-like of items to process
    return [str(item).lower() for item in (fallback if isinstance(fallback, list) else [fallback]) if item is not None and str(item).strip()]


def get_strat_param(params_dict: Dict, key: str, fallback, value_type: type = float):
    """Safely get a strategy parameter from the dictionary, converting to the specified type."""
    try:
        value_str = params_dict.get(key)
        if value_str is None:
            # logger.debug(f"Strategy param '{key}' not found in provided dict. Using fallback: {fallback}")
            return fallback
        # Handle boolean conversion explicitly
        if value_type == bool and isinstance(value_str, str):
             return value_str.lower() in ('yes', 'true', 't', '1')
        return value_type(value_str)
    except (ValueError, TypeError):
        logger.error(f"Failed to convert strategy param '{key}' value '{value_str}' to {value_type}. Using fallback: {fallback}")
        return fallback


def calculate_fib_tp_sl_prices(entry_price: float, is_long: bool, sl_distance_percent: float, tp_sl_ratio: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Calcule les prix de TP et SL basés sur des niveaux Fibonacci en fonction de la position.
    
    Args:
        entry_price: Prix d'entrée de la position
        is_long: True si position longue, False si position courte
        sl_distance_percent: Distance du SL en pourcentage du prix d'entrée (ex: 0.015 pour 1.5%)
        tp_sl_ratio: Ratio TP/SL (ex: 1.618 pour ratio Fibonacci PHI). Si None, ratio par défaut PHI utilisé.
    
    Returns:
        Tuple (tp_price, sl_price). Chacun peut être None si non calculable.
    """
    if entry_price <= 0 or sl_distance_percent <= 0:
        return None, None

    # Appliquer le ratio par défaut PHI si non spécifié
    if tp_sl_ratio is None or tp_sl_ratio <= 0:
        tp_sl_ratio = indicators.PHI  # The golden ratio (≈1.618), from indicators module

    # Pour position longue
    if is_long:
        sl_price = entry_price * (1 - sl_distance_percent)
        sl_distance_abs = entry_price - sl_price
        tp_distance_abs = sl_distance_abs * tp_sl_ratio
        tp_price = entry_price + tp_distance_abs
    # Pour position courte (non implémenté actuellement)
    else:
        sl_price = entry_price * (1 + sl_distance_percent)
        sl_distance_abs = sl_price - entry_price
        tp_distance_abs = sl_distance_abs * tp_sl_ratio
        tp_price = entry_price - tp_distance_abs

    return tp_price, sl_price


# --- Backtest Metrics Calculation (from backtest_metrics.py) ---

def calculate_performance_metrics(trades_df: pd.DataFrame, days_df: pd.DataFrame, annualization_factor: float = 252) -> Dict:
    """
    Calcule un ensemble complet de métriques de performance pour un backtest.
    Args:
        trades_df: DataFrame des trades (doit contenir 'profit_usd', 'open_date', 'close_date').
        days_df: DataFrame journalier (doit contenir 'wallet', index DatetimeIndex).
        annualization_factor: Facteur pour l'annualisation (ex: 252 pour jours de bourse, 365 pour jours calendaires).
    Returns:
        Un dictionnaire de métriques.
    """
    metrics = {}

    if days_df is None or days_df.empty:
        logger.warning("DataFrame journalier vide. Impossible de calculer les métriques basées sur le wallet.")
        # Add basic metrics if trades_df is available
        if trades_df is not None and not trades_df.empty:
             metrics['total_trades'] = len(trades_df)
             # Add basic trade counts if profit_usd exists
             if 'profit_usd' in trades_df.columns:
                  metrics['winning_trades'] = len(trades_df[trades_df['profit_usd'] > 0])
                  metrics['losing_trades'] = len(trades_df[trades_df['profit_usd'] < 0])
                  metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0.0
             else:
                  metrics['winning_trades'] = 0; metrics['losing_trades'] = 0; metrics['win_rate'] = 0.0

        else:
             metrics['total_trades'] = 0

        # Fill remaining metrics with NaN
        metrics['total_return_percent'] = np.nan
        metrics['sharpe_ratio'] = np.nan
        metrics['max_drawdown_percent'] = np.nan
        metrics['avg_profit_per_trade_usd'] = np.nan
        metrics['avg_loss_per_trade_usd'] = np.nan
        metrics['profit_factor'] = np.nan
        metrics['avg_trade_duration_seconds'] = np.nan
        metrics['max_winning_trade_usd'] = np.nan
        metrics['max_losing_trade_usd'] = np.nan


        return metrics


    # Ensure that the 'wallet' column is numeric
    if not pd.api.types.is_numeric_dtype(days_df['wallet']):
        logger.error("La colonne 'wallet' dans days_df n'est pas numérique.")
        # Attempt coercion, or return empty metrics
        try:
            days_df['wallet'] = pd.to_numeric(days_df['wallet'])
            logger.warning("La colonne 'wallet' a été convertie en numérique.")
        except Exception as e:
            logger.error(f"Impossible de convertir la colonne 'wallet' en numérique: {e}")
            return metrics # Return empty metrics if critical conversion fails


    # Ensure index is DatetimeIndex for daily return calculation
    if not isinstance(days_df.index, pd.DatetimeIndex):
        logger.error("L'index du DataFrame journalier n'est pas un DatetimeIndex.")
        # Attempt conversion
        try:
            days_df.index = pd.to_datetime(days_df.index)
            logger.warning("Index du DataFrame journalier converti en DatetimeIndex.")
        except Exception as e:
            logger.error(f"Impossible de convertir l'index du DataFrame journalier en DatetimeIndex: {e}")
            return metrics # Return empty metrics if essential index conversion fails


    initial_capital = days_df['wallet'].iloc[0]
    final_capital = days_df['wallet'].iloc[-1]
    metrics['total_return_percent'] = float((final_capital - initial_capital) / initial_capital) * 100.0 # Ensure float

    # Calculate daily returns
    days_df['daily_return'] = days_df['wallet'].pct_change().fillna(0)

    # Sharpe Ratio (annualized)
    mean_daily_return = days_df['daily_return'].mean()
    std_daily_return = days_df['daily_return'].std()
    if std_daily_return != 0 and not np.isnan(std_daily_return):
        sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(annualization_factor)
        metrics['sharpe_ratio'] = float(sharpe_ratio)
    else:
        metrics['sharpe_ratio'] = np.nan

    # Maximum Drawdown
    days_df['peak_wallet'] = days_df['wallet'].cummax()
    days_df['drawdown'] = days_df['peak_wallet'] - days_df['wallet']
    days_df['drawdown_percent'] = days_df['drawdown'] / days_df['peak_wallet']
    max_drawdown_percent = days_df['drawdown_percent'].max()
    metrics['max_drawdown_percent'] = float(max_drawdown_percent) * 100.0 if not np.isnan(max_drawdown_percent) else np.nan # Convert to percentage

    # Metrics based on trades
    if trades_df is None or trades_df.empty:
        metrics['total_trades'] = 0
        metrics['winning_trades'] = 0
        metrics['losing_trades'] = 0
        metrics['win_rate'] = 0.0
        metrics['avg_profit_per_trade_usd'] = 0.0
        metrics['avg_loss_per_trade_usd'] = 0.0
        metrics['profit_factor'] = np.nan
        metrics['avg_trade_duration_seconds'] = np.nan
        metrics['max_winning_trade_usd'] = 0.0
        metrics['max_losing_trade_usd'] = 0.0

    else:
        # Ensure 'profit_usd' is numeric
        if not pd.api.types.is_numeric_dtype(trades_df['profit_usd']):
             logger.warning("Colonne 'profit_usd' dans trades_df n'est pas numérique. Tentative de conversion.")
             try: trades_df['profit_usd'] = pd.to_numeric(trades_df['profit_usd'])
             except Exception as e:
                  logger.error(f"Impossible de convertir la colonne 'profit_usd': {e}. Métriques basées sur profit limitées.")
                  # Return partial metrics if conversion fails
                  metrics['total_trades'] = len(trades_df)
                  return metrics | {
                      'winning_trades': np.nan, 'losing_trades': np.nan, 'win_rate': np.nan,
                      'avg_profit_per_trade_usd': np.nan, 'avg_loss_per_trade_usd': np.nan, 'profit_factor': np.nan,
                      'avg_trade_duration_seconds': np.nan, 'max_winning_trade_usd': np.nan, 'max_losing_trade_usd': np.nan
                  }


        metrics['total_trades'] = len(trades_df)
        winning_trades_df = trades_df[trades_df['profit_usd'] > 1e-9] # Use tolerance for float comparison
        losing_trades_df = trades_df[trades_df['profit_usd'] < -1e-9]
        metrics['winning_trades'] = len(winning_trades_df)
        metrics['losing_trades'] = len(losing_trades_df)

        metrics['win_rate'] = float(metrics['winning_trades'] / metrics['total_trades']) if metrics['total_trades'] > 0 else 0.0

        metrics['avg_profit_per_trade_usd'] = float(winning_trades_df['profit_usd'].mean()) if metrics['winning_trades'] > 0 else 0.0
        metrics['avg_loss_per_trade_usd'] = float(losing_trades_df['profit_usd'].mean()) if metrics['losing_trades'] > 0 else 0.0 # Will be negative

        total_profit_usd = winning_trades_df['profit_usd'].sum()
        total_loss_usd = losing_trades_df['profit_usd'].sum()
        metrics['profit_factor'] = float(total_profit_usd / abs(total_loss_usd)) if total_loss_usd < -1e-9 else (float('inf') if total_profit_usd > 1e-9 else np.nan)

        # Average trade duration
        if 'open_date' in trades_df.columns and 'close_date' in trades_df.columns:
             try:
                  # Ensure date columns are datetime
                  trades_df['open_date'] = pd.to_datetime(trades_df['open_date'])
                  trades_df['close_date'] = pd.to_datetime(trades_df['close_date'])
                  trades_df['duration'] = (trades_df['close_date'] - trades_df['open_date']).dt.total_seconds()
                  metrics['avg_trade_duration_seconds'] = float(trades_df['duration'].mean()) if metrics['total_trades'] > 0 else np.nan
             except Exception as e:
                  logger.warning(f"Impossible de calculer la durée moyenne des trades: {e}")
                  metrics['avg_trade_duration_seconds'] = np.nan
        else:
            metrics['avg_trade_duration_seconds'] = np.nan

        # Maximum profit/loss trade
        metrics['max_winning_trade_usd'] = float(winning_trades_df['profit_usd'].max()) if metrics['winning_trades'] > 0 else 0.0
        metrics['max_losing_trade_usd'] = float(losing_trades_df['profit_usd'].min()) if metrics['losing_trades'] > 0 else 0.0 # Min will be the largest negative


    return metrics


# --- Backtester Core (from backtester.py, adapted) ---

# Type hint for internal position tracking
BacktestPosition = Dict[str, float | str | datetime | None | bool | pd.Timestamp]
BacktestPositions = Dict[str, BacktestPosition]


class Backtester:
    def __init__(self,
                 prepared_data: Dict[str, pd.DataFrame],
                 config: ConfigParser,
                 loaded_models: Dict,
                 strategy_params: Dict
                 ):
        """
        Initialise le backtester.
        Args:
            prepared_data: Dictionnaire {symbol_market: DataFrame complet} avec toutes les données nécessaires.
            config: L'objet ConfigParser (pour les params backtest et trading).
            loaded_models: Dictionnaire avec les modèles ML, scaler et paramètres d'entraînement.
            strategy_params: Dictionnaire des paramètres de stratégie (strings de config ou perturbés).
        """
        if not prepared_data: raise ValueError("prepared_data dictionary is empty.")
        self.prepared_data = prepared_data
        self.config = config
        self.loaded_models = loaded_models
        self.strategy_params = strategy_params # Store the provided strategy params dict

        # Load ML components and training params
        self.xgb_model = loaded_models.get('xgb')
        self.fq_model = loaded_models.get('fq')
        self.feature_scaler = loaded_models.get('scaler')
        self.training_params = loaded_models.get('params', {})
        self.xgb_features_names = self.training_params.get('xgboost_features', [])
        self.fq_quantiles = self.training_params.get('futurequant_quantiles', [])
        self.all_feature_columns = self.training_params.get('feature_columns', [])

        # Check if essential ML components/params are loaded
        self.ml_ready = True
        if self.xgb_model is None or self.fq_model is None or self.feature_scaler is None or not self.xgb_features_names or not self.all_feature_columns or not self.fq_quantiles:
             logger.warning("Modèles ML, scaler ou paramètres d'entraînement manquants/incomplets. ML prediction will be skipped.")
             self.ml_ready = False
             # Check if strategy params exist even if ML is not ready
             if not self.strategy_params:
                  logger.error("Strategy parameters not provided. Backtest will likely fail.")


        # Get backtest configuration from self.config (NOT strategy_params)
        try:
            self.initial_capital = get_config_option(config, 'backtest', 'initial_capital', fallback=10000, value_type=float)
            self.taker_fee = get_config_option(config, 'backtest', 'taker_fee', fallback=0.0006, value_type=float)
            self.maker_fee = get_config_option(config, 'backtest', 'maker_fee', fallback=0.0002, value_type=float)
            self.slippage_percent = get_config_option(config, 'backtest', 'slippage_percent', fallback=0.0001, value_type=float)
            self.max_concurrent_positions = get_config_option(config, 'trading', 'max_concurrent_positions', fallback=5, value_type=int)

            # Get column names from config
            self.ts_col = get_config_option(config, 'backtest', 'timestamp_col', fallback='timestamp', value_type=str)
            self.open_col = get_config_option(config, 'backtest', 'open_col', fallback='open', value_type=str)
            self.high_col = get_config_option(config, 'backtest', 'high_col', fallback='high', value_type=str)
            self.low_col = get_config_option(config, 'backtest', 'low_col', fallback='low', value_type=str)
            self.close_col = get_config_option(config, 'backtest', 'close_col', fallback='close', value_type=str)
            self.volume_col = get_config_option(config, 'backtest', 'volume_col', fallback='volume', value_type=str)
            self.rh_col = get_config_option(config, 'backtest', 'rh_col', fallback='R_H', value_type=str)
            self.ra_col = get_config_option(config, 'backtest', 'ra_col', fallback='R_A', value_type=str)
            self.sentiment_col = get_config_option(config, 'backtest', 'sentiment_col', fallback='sentiment_score', value_type=str)

        except Exception as e:
            logger.error(f"Erreur de configuration [backtest] ou [trading]: {e}. Utilisation des fallbacks.", exc_info=True)
            # Minimal defaults if config fails
            self.initial_capital = 10000; self.taker_fee = 0.0006; self.maker_fee = 0.0002; self.slippage_percent = 0.0001; self.max_concurrent_positions = 5
            self.ts_col = 'timestamp'; self.open_col = 'open'; self.high_col = 'high'; self.low_col = 'low'; self.close_col = 'close'; self.volume_col = 'volume'
            self.rh_col = 'R_H'; self.ra_col = 'R_A'; self.sentiment_col = 'sentiment_score'


        # Load DRM and Exit parameters from self.strategy_params dict (they come from config or perturbation)
        # strategy_hybrid will use these parameters from the dict passed to it.
        # We need some of them here in Backtester for exit level updates and sizing calculation.
        # Use get_strat_param helper
        try:
             self.move_to_be_profit_percent = get_strat_param(strategy_params, 'move_to_be_profit_percent', 0.005, float)
             self.trailing_stop_profit_percent_start = get_strat_param(strategy_params, 'trailing_stop_profit_percent_start', 0.01, float)
             self.trailing_stop_distance_percent = get_strat_param(strategy_params, 'trailing_stop_distance_percent', 0.005, float)
             # Daily PnL adjustment config string is needed by _parse_daily_pnl_adjustment
             daily_pnl_adj_str = get_strat_param(strategy_params, 'daily_pnl_risk_adjustment', '', str)
             self.daily_pnl_risk_adjustment_tiers = self._parse_daily_pnl_adjustment(daily_pnl_adj_str)

             self.min_risk_multiplier = get_strat_param(strategy_params, 'min_risk_multiplier', 0.1, float)
             # Session risk multipliers are needed by _get_session_multiplier
             self.session_risk_multipliers = {
                  'asia': get_strat_param(strategy_params, 'session_risk_multiplier_asia', 1.0, float),
                  'london': get_strat_param(strategy_params, 'session_risk_multiplier_london', 1.0, float),
                  'ny': get_strat_param(strategy_params, 'session_risk_multiplier_ny', 1.2, float),
                  'london_ny_overlap': get_strat_param(strategy_params, 'session_risk_multiplier_london_ny_overlap', 1.1, float) # Example overlap multiplier
                  # Add other sessions from config and parse them here
             }


        except Exception as e:
            logger.error(f"Erreur lors du chargement des paramètres de stratégie nécessaires au Backtester: {e}. Utilisation des fallbacks.", exc_info=True)
            self.move_to_be_profit_percent = 0.005; self.trailing_stop_profit_percent_start = 0.01; self.trailing_stop_distance_percent = 0.005; self.daily_pnl_risk_adjustment_tiers = [(float('-inf'), 1.0)]; self.min_risk_multiplier = 0.1
            self.session_risk_multipliers = {'asia': 1.0, 'london': 1.0, 'ny': 1.0, 'london_ny_overlap': 1.0}


        # Find the time index and start point
        self.time_index = None
        min_start_date = pd.Timestamp.max
        valid_pairs = 0
        for pair, df in self.prepared_data.items():
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                valid_pairs += 1
                # Ensure required columns exist for each pair's data
                required_cols_base = [self.open_col, self.high_col, self.low_col, self.close_col, self.volume_col, self.rh_col, self.ra_col, self.sentiment_col]
                required_cols_ml = self.all_feature_columns if self.ml_ready else [] # Need all features if ML is used

                missing_pair_cols = set(required_cols_base + required_cols_ml) - set(df.columns)
                if missing_pair_cols:
                     logger.warning(f"Données préparées pour {pair} manquent des colonnes requises: {missing_pair_cols}. Ce pair sera traité avec des limitations ou sautera.")
                     # Handle missing columns: fill with 0 or NaN, or skip the pair.
                     # For backtest, skipping the pair if essential data is missing is safer.
                     continue # Skip this pair if essential columns are missing (e.g., OHLCV, Ratings, Sentiment)
                     # For ML features, if self.all_feature_columns are missing, ml_ready will be False, which is handled.


                min_start_date = min(min_start_date, df.index.min())
                if self.time_index is None: self.time_index = df.index # Use first valid pair's index as base

        if valid_pairs == 0 or self.time_index is None or min_start_date == pd.Timestamp.max:
             raise ValueError("prepared_data is empty, has no valid pairs, or could not establish a time index.")

        # Ensure the main time_index is sorted
        self.time_index = self.time_index.sort_values()

        # Determine the first valid index to start the loop >= min_start_date
        start_idx_in_time_index = self.time_index.get_loc(min_start_date, method='bfill')

        logger.info(f"Backtester initialized. Running backtest from {self.time_index[start_idx_in_time_index]} to {self.time_index[-1]}.")

        # Pre-calculate ATR series for each asset
        self.atr_series_map = {}
        atr_period = self.config.getint('strategy_hybrid', 'atr_period_sl_tp', fallback=14)
        for symbol_market, df in self.prepared_data.items():
            if df is not None and not df.empty and \
               all(col in df.columns for col in ['high', 'low', 'close']):
                self.atr_series_map[symbol_market] = calculate_atr(
                    df['high'],
                    df['low'],
                    df['close'],
                    time_period=atr_period
                )
            else:
                self.atr_series_map[symbol_market] = pd.Series(dtype=float)


    def _parse_daily_pnl_adjustment(self, config_str: str) -> List[Tuple[float, float]]:
        """
        Parses the daily PnL adjustment tiers from config string.
        Format: "threshold1=multiplier1, threshold2=multiplier2, ..."
        Example: "-500=0.5, 0=1.0, 1000=1.5"
        
        Returns a list of (threshold, multiplier) tuples sorted by threshold ascending.
        """
        try:
            if not config_str or config_str.strip() == '':
                # Default: no adjustment
                return [(float('-inf'), 1.0)]
            
            tiers = []
            
            # Split by comma and parse each threshold=multiplier pair
            for pair in config_str.split(','):
                if '=' not in pair:
                    logger.warning(f"Invalid daily PnL adjustment format: '{pair}'. Expected 'threshold=multiplier'.")
                    continue
                    
                threshold_str, multiplier_str = pair.split('=')
                try:
                    threshold = float(threshold_str.strip())
                    multiplier = float(multiplier_str.strip())
                    tiers.append((threshold, multiplier))
                except ValueError:
                    logger.warning(f"Invalid threshold or multiplier in '{pair}'. Using default.")
                    continue
            
            # Ensure at least one tier exists
            if not tiers:
                return [(float('-inf'), 1.0)]
                
            # Sort by threshold (ascending)
            return sorted(tiers, key=lambda x: x[0])
            
        except Exception as e:
            logger.error(f"Error parsing daily PnL adjustment config: {e}. Using default (no adjustment).")
            return [(float('-inf'), 1.0)]

    def _get_daily_pnl_multiplier(self, today_pnl: float) -> float:
        """
        Gets the dynamic risk multiplier based on daily PnL.
        Uses the daily_pnl_risk_adjustment_tiers to find the appropriate multiplier.
        """
        # Find the appropriate tier for current daily PnL
        multiplier = 1.0  # Default if no tiers match
        
        for threshold, tier_multiplier in self.daily_pnl_risk_adjustment_tiers:
            if today_pnl >= threshold:
                multiplier = tier_multiplier
            else:
                break  # Tiers are sorted, stop when we find the first threshold we're below
        
        return multiplier

    def _get_session_multiplier(self, current_ts: datetime) -> float:
        """Gets the session risk multiplier based on the current timestamp."""
        # Use current_ts directly, assuming it's timezone-aware (UTC from data prep)
        current_hour_utc = current_ts.hour

        # Use multiplier mapping from self.session_risk_multipliers
        session = 'asia' # Default
        if 7 <= current_hour_utc < 12: session = 'london'
        elif 12 <= current_hour_utc < 16: session = 'london_ny_overlap'
        elif 16 <= current_hour_utc < 21: session = 'ny'
        elif 21 <= current_hour_utc or current_hour_utc < 7: session = 'asia'

        return self.session_risk_multipliers.get(session, 1.0) # Use multiplier from loaded dict

    def run_backtest(self) -> Dict:
        """
        Executes the backtest simulation on the historical prepared data.
        Handles updated strategy logic, DRM, and exit strategies.
        """
        logger.info("="*40)
        logger.info(" DÉBUT DU BACKTEST ")
        logger.info(f"Capital initial: {self.initial_capital}")

        # --- Initialisation du Backtest ---
        wallet = self.initial_capital
        current_positions: BacktestPositions = {}
        trades = []
        days = []

        # DRM state tracking
        current_day_ts = None # Keep track of the timestamp of the first bar of the current day
        today_pnl = 0.0 # PnL accumulated since the start of the current day's data loop


        start_idx_in_time_index = self.time_index.get_loc(self.time_index.min(), method='bfill')

        # Loop through time steps (bar by bar)
        for i in tqdm(range(start_idx_in_time_index, len(self.time_index) - 1), desc="Running Backtest"):
            current_ts = self.time_index[i] # Decision point time ('t')
            next_ts = self.time_index[i+1] # Execution/Check time ('t+1')

            # --- Gestion Journalière (pour DRM et rapport) ---
            if current_day_ts is None or current_ts.date() != current_day_ts.date():
                 if current_day_ts is not None: # If not the very first bar
                      # Record end-of-day wallet including open positions value at the close of the bar just finished (current_ts)
                      end_of_day_portfolio_value = wallet
                      for pair, pos_data in current_positions.items():
                           pair_df = self.prepared_data.get(pair)
                           if pair_df is not None and current_ts in pair_df.index:
                                current_close_price = pair_df.loc[current_ts, self.close_col]
                                if pos_data['side'] == 'long':
                                     end_of_day_portfolio_value += pos_data['qty'] * current_close_price

                      days.append({
                          'timestamp': current_day_ts.normalize(), # Date of the day that just ended
                          'wallet': end_of_day_portfolio_value,
                      })
                      # logger.debug(f"End of Day {current_day_ts.date()}. Wallet: {end_of_day_portfolio_value:.2f}, Today PnL: {today_pnl:.2f}") # Too verbose

                 # Start a new day
                 current_day_ts = current_ts # Mark the timestamp of the first bar of the new day
                 today_pnl = 0.0 # Reset daily PnL


            # --- Collecter Inputs & Make Decisions for each Actif ---
            latest_prices_t: Dict[str, float] = {}
            next_bar_data: Dict[str, pd.Series] = {}
            asset_decisions: Dict[str, Dict] = {}

            for symbol_market, pair_df in self.prepared_data.items():
                 if current_ts in pair_df.index and next_ts in pair_df.index:
                      current_bar = pair_df.loc[current_ts] # Row at time 't' with all features
                      next_bar = pair_df.loc[next_ts]     # Row at time 't+1' (OHLCV for simulation)

                      current_price_t = current_bar[self.close_col]
                      latest_prices_t[symbol_market] = current_price_t
                      next_bar_data[symbol_market] = next_bar

                      # Check if essential base columns exist in current_bar
                      required_base_cols_exist = all(col in current_bar for col in [self.close_col, self.rh_col, self.ra_col, self.sentiment_col] + [k for k in self.all_feature_columns if not (k.startswith('ms_') or k.startswith('of_'))])
                      if not required_base_cols_exist:
                           logger.warning(f"Colonnes de features de base manquantes pour {symbol_market} à {current_ts}. Skip décision pour cet actif.")
                           continue # Skip decision for this asset if base features missing

                      # Get inputs from current_bar
                      current_ratings = {self.rh_col: current_bar[self.rh_col], self.ra_col: current_bar[self.ra_col]}
                      sentiment_score_t = current_bar[self.sentiment_col]
                      ms_features_t = {k: current_bar[k] for k in self.all_feature_columns if k.startswith('ms_')}
                      of_features_t = {k: current_bar[k] for k in self.all_feature_columns if k.startswith('of_')}


                      # Prepare ML Inputs and Run Predictions (only if ml_ready)
                      xgb_probabilities = None
                      fq_predicted_ratios = None

                      if self.ml_ready:
                           # XGBoost
                           xgb_features_names = self.loaded_models.get('params', {}).get('xgboost_features', [])
                           all_feature_columns_list = self.loaded_models.get('params', {}).get('feature_columns', [])
                           if xgb_features_names and all_feature_columns_list and self.feature_scaler:
                                complete_raw_features_t = feature_engineer.build_complete_raw_features(current_bar, ms_features_t, of_features_t) # This adds R_Diff/R_Ratio
                                if complete_raw_features_t is not None:
                                     # Select and scale for XGBoost
                                     xgb_features_t_aligned = complete_raw_features_t[xgb_features_names]
                                     # Need to ensure order is correct for scaler and model!
                                     # Order from training_params['xgboost_features'] is the correct order.
                                     xgb_features_t_aligned_ordered = xgb_features_t_aligned[self.xgb_features_names]

                                     xgb_features_t_scaled_np = self.feature_scaler.transform(xgb_features_t_aligned_ordered.values.reshape(1, -1))
                                     xgb_features_t_scaled_df = pd.DataFrame(xgb_features_t_scaled_np, columns=self.xgb_features_names)
                                     if not xgb_features_t_scaled_df.isnull().values.any().any(): # Check if any NaN
                                          xgb_probabilities = model_xgboost.predict_xgboost(xgb_features_t_scaled_df)
                                     # else: logger.warning(f"NaNs in scaled XGB features for {symbol_market} at {current_ts}") # Too verbose
                           # else: logger.debug("XGBoost prediction requirements not met.") # Too verbose

                           # FutureQuant
                           if self.fq_model and all_feature_columns_list and self.feature_scaler:
                                X_live_fq_seq = feature_engineer.build_feature_sequence_for_fq(
                                     pair_df, self.config['futurequant'], self.feature_scaler, all_feature_columns_list, current_ts
                                )
                                if X_live_fq_seq is not None:
                                     fq_predicted_ratios = model_futurequant.predict_futurequant(X_live_fq_seq)
                                # else: logger.debug(f"FQ sequence building requirements not met for {symbol_market} at {current_ts}") # Too verbose
                           # else: logger.debug("FutureQuant prediction requirements not met.") # Too verbose


                      # Get Current Position State
                      current_position_state = current_positions.get(symbol_market)
                      current_position_qty = current_position_state['qty'] if current_position_state else 0.0

                      # Retrieve ATR value for current_ts and symbol_market
                      atr_value_for_decision = None
                      asset_specific_atr_series = self.atr_series_map.get(symbol_market)
                      if asset_specific_atr_series is not None and not asset_specific_atr_series.empty:
                          atr_value_for_decision = asset_specific_atr_series.get(current_ts)
                          if pd.isna(atr_value_for_decision):
                              # Fallback to the last valid ATR at or before current_ts
                              relevant_atrs_up_to_current_ts = asset_specific_atr_series.loc[:current_ts].dropna()
                              if not relevant_atrs_up_to_current_ts.empty:
                                  atr_value_for_decision = relevant_atrs_up_to_current_ts.iloc[-1]
                                  # logger.debug(f"[{symbol_market}] ATR for {current_ts} is NaN. Using last valid ATR: {atr_value_for_decision:.4f}") # Potentially too verbose
                              else:
                                  # logger.warning(f"[{symbol_market}] ATR for {current_ts} is NaN and no prior valid ATR found.") # Potentially too verbose
                                  atr_value_for_decision = None # Explicitly set to None
                      # else: logger.warning(f"[{symbol_market}] No pre-calculated ATR series found or series is empty for TradingAnalyzer.") # Potentially too verbose

                      # Generate Decision using the hybrid strategy
                      trade_decision = strategy_hybrid.generate_trade_decision(
                          symbol_market, current_ratings, current_price_t, sentiment_score_t,
                          xgb_probabilities, fq_predicted_ratios, ms_features_t, of_features_t,
                          atr_value_for_decision, # <<< Added atr_value_for_decision
                          current_position_qty, wallet, today_pnl, config=self.config
                      )
                      asset_decisions[symbol_market] = trade_decision

                 # else: logger.debug(f"Données manquantes pour {symbol_market} à {current_ts} ou {next_ts}. Skip décision.") # Too verbose


            # --- Process Decisions and Simulate Trades (at next_ts Open/during bar) ---
            # 0. Update Trailing Stops and Move to Break-Even using the CLOSE price of the *current_ts* bar
            updated_positions = {}
            for symbol_market, pos_data in current_positions.items():
                 if symbol_market in latest_prices_t: # If we have the last price at 't'
                      current_price = latest_prices_t[symbol_market]
                      updated_pos_data = self._update_position_exit_levels(pos_data, current_price, self.strategy_params)
                      updated_positions[symbol_market] = updated_pos_data
                 else:
                      updated_positions[symbol_market] = pos_data # Keep old levels


            # 1. Handle position closures (SL/TP or Signal) - Check using next_ts OHLC
            positions_to_close: List[str] = []
            for symbol_market, pos_data in list(updated_positions.items()): # Iterate over a copy
                 if symbol_market in next_bar_data: # Use next_bar_data collected earlier
                      next_bar = next_bar_data[symbol_market]
                      decision = asset_decisions.get(symbol_market) # Decision from current_ts

                      close_reason = None
                      close_price = None

                      if pos_data['side'] == 'long':
                          # Check SL/TP hit using next_bar OHLC vs updated SL/TP in pos_data
                          if pos_data['sl'] is not None and next_bar[self.low_col] <= pos_data['sl']:
                              close_reason = 'SL'
                              close_price = pos_data['sl']
                              if pos_data['tp'] is not None and next_bar[self.high_col] >= pos_data['tp']:
                                   logger.debug(f"SL and TP hit in same bar ({next_ts}) for {symbol_market}. SL takes precedence.")
                              close_price = max(next_bar[self.low_col], min(next_bar[self.high_col], close_price)) # Clamp to bar range
                          elif pos_data['tp'] is not None and next_bar[self.high_col] >= pos_data['tp']:
                              close_reason = 'TP'
                              close_price = pos_data['tp']
                              close_price = max(next_bar[self.low_col], min(next_bar[self.high_col], close_price)) # Clamp

                      # Check for Signal Close if SL/TP not hit
                      if close_reason is None and decision and decision['action'] == 'SELL_TO_CLOSE_LONG':
                          close_reason = 'Signal'
                          close_price = next_bar[self.open_col] # Simulate at next bar open


                      if close_reason is not None and close_price is not None:
                          # Simulate closing trade, update wallet and today_pnl
                          simulated_exit_price = close_price * (1 - self.slippage_percent if pos_data['side'] == 'long' else 1 + self.slippage_percent)
                          exit_fee = abs(pos_data['qty'] * simulated_exit_price) * self.taker_fee
                          profit_loss_usd = (simulated_exit_price - pos_data['entry_price']) * pos_data['qty'] # For long
                          net_pnl = profit_loss_usd - exit_fee

                          wallet += net_pnl
                          today_pnl += net_pnl # Update daily PnL

                          # Record trade
                          trades.append({
                              'pair': symbol_market, 'side': pos_data['side'], 'open_date': pos_data['open_date'], 'close_date': next_ts,
                              'entry_price': pos_data['entry_price'], 'exit_price': simulated_exit_price, 'qty': pos_data['qty'],
                              'profit_usd': net_pnl, 'total_fees_usd': pos_data['fees_paid'] + exit_fee,
                              'open_reason': pos_data['open_reason'], 'close_reason': close_reason,
                              'entry_wallet': pos_data['entry_wallet'], 'exit_wallet': wallet
                          })
                          positions_to_close.append(symbol_market)
                          # logger.debug(f"Position {symbol_market} closed by {close_reason} at {next_ts}. Net PnL: {net_pnl:.2f}. Wallet: {wallet:.2f}. Today PnL: {today_pnl:.2f}") # Too verbose

            # Remove closed positions
            for pair in positions_to_close:
                 if pair in current_positions: del current_positions[pair]


            # 2. Handle position openings ('BUY' signals)
            open_positions_count = len(current_positions)
            for symbol_market, decision in asset_decisions.items():
                 if decision['action'] == 'BUY' and symbol_market not in current_positions and open_positions_count < self.max_concurrent_positions and symbol_market in next_bar_data:
                      next_bar = next_bar_data[symbol_market]
                      simulated_entry_price = next_bar[self.open_col] * (1 + self.slippage_percent) # Slippage for long entry

                      # --- Dynamic Position Sizing ---
                      setup_quality = decision.get('setup_quality', 'C')
                      base_multiplier = get_strat_param(self.strategy_params, f'setup_size_multiplier_{setup_quality.lower()}', 1.0, float)
                      session_multiplier = self._get_session_multiplier(current_ts)
                      pnl_adjustment_multiplier = self._get_daily_pnl_multiplier(today_pnl)

                      final_risk_multiplier = base_multiplier * session_multiplier * pnl_adjustment_multiplier
                      min_risk_multiplier = get_strat_param(self.strategy_params, 'min_risk_multiplier', 0.1, float)
                      final_risk_multiplier = max(min_risk_multiplier, final_risk_multiplier)

                      base_position_value_percent = get_strat_param(self.config.sections().get('trading', {}), 'position_size_percent', 0.02, float)
                      target_position_value = wallet * base_position_value_percent * final_risk_multiplier


                      if simulated_entry_price > 0 and target_position_value > 0:
                           qty = target_position_value / simulated_entry_price
                           # Check min_qty from exchange if available (requires passing exchange info)

                           entry_fee = qty * simulated_entry_price * self.taker_fee

                           if wallet >= entry_fee:
                                wallet -= entry_fee # Deduct fees

                                # Calculate initial TP/SL prices
                                tp_price, sl_price = calculate_fib_tp_sl_prices(
                                    simulated_entry_price, True,
                                    get_strat_param(self.strategy_params, 'sl_distance_percent', 0.015, float),
                                    get_strat_param(self.strategy_params, 'tp_sl_ratio', indicators.PHI, float)
                                )

                                # Add position to tracking
                                current_positions[symbol_market] = {
                                    'qty': qty, 'entry_price': simulated_entry_price, 'side': 'long',
                                    'tp': tp_price, 'sl': sl_price, 'open_date': next_ts, 'fees_paid': entry_fee,
                                    'open_reason': decision.get('setup_quality', 'Signal'),
                                    'entry_wallet': wallet + entry_fee,
                                    'moved_to_be': False,
                                    'trailing_sl_active': False,
                                    'highest_price_since_entry': simulated_entry_price
                                }
                                open_positions_count += 1

                                logger.debug(f"Position LONG {symbol_market} opened at {next_ts}. Price: {simulated_entry_price:.4f}. Qty: {qty:.8f}. Wallet: {wallet:.2f}. Setup: {setup_quality}, Risk Multi: {final_risk_multiplier:.2f}")
                           else:
                                logger.debug(f"Fonds insuffisants ({wallet:.2f}) pour couvrir frais ({entry_fee:.2f}) pour {symbol_market} à {next_ts}. Non exécuté.")
                        # else: logger.debug(f"Quantité cible ({target_position_value:.2f} USD) ou prix ({simulated_entry_price:.4f}) invalide.") # Too verbose


            # --- Clôturer les positions ouvertes à la fin du backtest ---
        logger.info(f"\nFin des données. Clôture de {len(current_positions)} positions ouvertes restantes...")
        # Use the very last timestamp from the time_index
        if len(self.time_index) > start_idx_in_time_index: # If backtest ran for at least one bar
             last_bar_ts = self.time_index.iloc[-1] # Use iloc[-1] for the last timestamp in the index

             for symbol_market, pos_data in list(current_positions.items()):
                 # Use the close price of the very last bar as the exit price (simplification)
                 pair_df = self.prepared_data.get(symbol_market)
                 if pair_df is not None and last_bar_ts in pair_df.index:
                      simulated_exit_price = pair_df.loc[last_bar_ts, self.close_col]

                      # Simulate slippage
                      simulated_exit_price *= (1 - self.slippage_percent if pos_data['side'] == 'long' else 1 + self.slippage_percent)

                      # Fees on exit
                      exit_fee = abs(pos_data['qty'] * simulated_exit_price) * self.taker_fee

                      profit_loss_usd = (simulated_exit_price - pos_data['entry_price']) * pos_data['qty'] # For long
                      net_pnl = profit_loss_usd - exit_fee

                      wallet += net_pnl
                      today_pnl += net_pnl # Update daily PnL for the last day

                      trades.append({
                          'pair': symbol_market, 'side': pos_data['side'], 'open_date': pos_data['open_date'], 'close_date': last_bar_ts,
                          'entry_price': pos_data['entry_price'], 'exit_price': simulated_exit_price, 'qty': pos_data['qty'],
                          'profit_usd': net_pnl, 'total_fees_usd': pos_data['fees_paid'] + exit_fee,
                          'open_reason': pos_data['open_reason'], 'close_reason': 'EndOfData',
                          'entry_wallet': pos_data['entry_wallet'], 'exit_wallet': wallet
                      })
                      # logger.debug(f"Position {symbol_market} closed at EndOfData ({last_bar_ts}). Net PnL: {net_pnl:.2f}. Final Wallet: {wallet:.2f}") # Too verbose

        # --- Finalisation et Métriques ---
        trades_df = pd.DataFrame(trades)
        days_df = pd.DataFrame(days) # Contains daily (or per-bar) snapshots

        # Add the final wallet value at the very last timestamp in days_df for full equity curve
        if len(self.time_index) > start_idx_in_time_index: # If backtest ran for at least one bar
             last_bar_ts = self.time_index.iloc[-1]
             final_day_date = last_bar_ts.normalize()

             # Find the last snapshot date in days_df
             last_snapshot_date = days_df.index.max() if not days_df.empty else None

             # Add final day snapshot if it's not already there (handles case where backtest ends exactly at day change)
             if last_snapshot_date is None or last_snapshot_date.normalize() != final_day_date:
                 # Recalculate final portfolio value just before adding the snapshot
                 final_portfolio_value = wallet
                 # Add value of remaining open positions (if any) at the very last close price
                 for pair, pos_data in current_positions.items(): # Should be empty if all closed at EndOfData
                      pair_df = self.prepared_data.get(pair)
                      if pair_df is not None and last_bar_ts in pair_df.index:
                           last_close_price = pair_df.loc[last_bar_ts, self.close_col]
                           if pos_data['side'] == 'long':
                              final_portfolio_value += pos_data['qty'] * last_close_price

                 days_df_final_row = pd.DataFrame([{'timestamp': final_day_date, 'wallet': final_portfolio_value}]).set_index('timestamp')
                 days_df = pd.concat([days_df, days_df_final_row])
                 days_df.sort_index(inplace=True) # Re-sort after adding


        elif trades_df.empty:
             # If no trades and no days recorded (e.g. data too short), add initial capital as only day
             days_df = pd.DataFrame([{'timestamp': self.time_index[start_idx_in_time_index].normalize(), 'wallet': self.initial_capital}]).set_index('timestamp')

        # Remove duplicate index entries if any
        days_df = days_df[~days_df.index.duplicated(keep='last')]


        # Calculate final metrics
        metrics = calculate_performance_metrics(trades_df, days_df)

        logger.info("="*40)
        logger.info(" FIN DU BACKTEST ")
        logger.info(f"Final Wallet Value: {days_df['wallet'].iloc[-1]:.2f}" if not days_df.empty else f"Final Cash: {wallet:.2f}")
        logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"Total Return: {metrics.get('total_return_percent', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', np.nan):.4f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown_percent', 0):.2f}%")
        logger.info(f"Profit Factor: {metrics.get('profit_factor', np.nan):.4f}")
        logger.info("="*40)

        return {
            'wallet_final_cash': wallet, # Cash remaining
            'final_portfolio_value': days_df['wallet'].iloc[-1] if not days_df.empty else wallet, # Total value
            'trades': trades_df,
            'days': days_df, # Daily (or per-bar) snapshots
            'metrics': metrics # Dictionary of calculated metrics
        }

    def _update_position_exit_levels(self, pos_data: BacktestPosition, current_price: float, strategy_params: Dict) -> BacktestPosition:
        """
        Updates SL/TP for a single position based on Move to BE and Trailing Stop rules.
        Uses the current price (at time 't') to check trigger conditions.
        Returns the updated pos_data dictionary.
        """
        if pos_data is None or pos_data['side'] != 'long' or current_price is None or current_price <= 0 or pos_data.get('entry_price') is None or pos_data['entry_price'] <= 0:
             return pos_data # No update needed

        updated_pos = pos_data.copy()
        entry_price = updated_pos['entry_price']
        initial_sl = get_strat_param(strategy_params, 'sl_distance_percent', 0.015, float) # Use initial SL distance from params

        # Calculate current profit percentage relative to entry
        current_profit_percent_relative_entry = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

        # --- 1. Check and Apply Move to Break-Even ---
        # Only if Move to BE hasn't happened yet
        if not updated_pos.get('moved_to_be', False):
             move_to_be_profit_percent = get_strat_param(strategy_params, 'move_to_be_profit_percent', 0.005, float)
             # Check if current price has reached or exceeded the BE trigger price
             if current_profit_percent_relative_entry >= move_to_be_profit_percent:
                  # Calculate Break Even price
                  fees_per_unit = updated_pos.get('fees_paid', 0.0) / updated_pos['qty'] if updated_pos.get('qty', 0) > 0 else 0.0
                  be_price = entry_price + fees_per_unit # BE price including fees

                  # Only move SL if the new BE price is higher than the current SL price
                  # The initial SL price was calculated based on sl_distance_percent from entry.
                  # We need to compare the BE price to the *current SL value* stored in pos_data.
                  current_sl_value = updated_pos.get('sl', float('-inf')) # Use -inf if SL is None initially

                  if be_price > current_sl_value:
                       updated_pos['sl'] = float(be_price)
                       updated_pos['moved_to_be'] = True
                       logger.debug(f"SL moved to Break Even ({updated_pos['sl']:.4f}) for {symbol_market} at price {current_price:.4f}.")


        # --- 2. Check and Apply Trailing Stop ---
        # Trailing stop activates once price is X% in profit relative to entry
        trailing_stop_profit_percent_start = get_strat_param(strategy_params, 'trailing_stop_profit_percent_start', 0.01, float)

        if current_profit_percent_relative_entry >= trailing_stop_profit_percent_start:
             updated_pos['trailing_sl_active'] = True # Mark TS as active


        # If Trailing Stop is active, update SL based on highest price reached
        if updated_pos.get('trailing_sl_active', False):
             # Update highest price reached since entry
             updated_pos['highest_price_since_entry'] = max(
                 updated_pos.get('highest_price_since_entry', entry_price), # Use entry_price as baseline if not set
                 current_price # Update with current price
             )

             # Calculate the new trailing stop price based on distance below highest price
             trailing_stop_distance_percent = get_strat_param(strategy_params, 'trailing_stop_distance_percent', 0.005, float)
             new_trailing_sl_price = updated_pos['highest_price_since_entry'] * (1 - trailing_stop_distance_percent)

             # Update SL only if the new trailing SL is higher than the current SL
             current_sl_value = updated_pos.get('sl', float('-inf'))
             if new_trailing_sl_price > current_sl_value:
                  updated_pos['sl'] = float(new_trailing_sl_price)
                  # No need to reset moved_to_be, Trailing Stop takes over SL management
                  logger.debug(f"Trailing SL updated to {updated_pos['sl']:.4f} for {symbol_market} at price {current_price:.4f}.")

        return updated_pos


# --- Robustness Analyzer (from robustness_analyzer.py, adapted) ---

class RobustnessAnalyzer:
    def __init__(self,
                 backtester_class: type,
                 base_strategy_params: Dict,
                 prepared_data: Dict[str, pd.DataFrame],
                 config: ConfigParser,
                 loaded_models: Dict
                 ):
        """
        Initialise l'analyseur de robustesse.
        Args:
            backtester_class: La classe Backtester à instancier pour chaque simulation.
            base_strategy_params: Le dictionnaire des paramètres de stratégie "optimaux" (strings).
            prepared_data: Les données historiques préparées {symbol_market: DataFrame}.
            config: L'objet ConfigParser (pour les configs backtest/robustesse).
            loaded_models: Dictionnaire des modèles ML entraînés, scaler, etc.
        """
        self.backtester_class = backtester_class
        self.base_strategy_params = base_strategy_params
        self.prepared_data = prepared_data
        self.config = config
        self.loaded_models = loaded_models

        # Get robustness configuration from self.config
        try:
            self.n_simulations = get_config_option(config, 'robustness', 'n_simulations', fallback=500, value_type=int)
            self.confidence_level = get_config_option(config, 'robustness', 'confidence_level', fallback=0.95, value_type=float)
            self.perturbation_range = get_config_option(config, 'robustness', 'perturbation_range', fallback=0.1, value_type=float)
            self.params_to_perturb_names = get_config_list_lower(config, 'robustness', 'parameters_to_perturb')

        except Exception as e:
            logger.error(f"Erreur de configuration [robustness]: {e}. Utilisation des fallbacks.", exc_info=True)
            self.n_simulations = 500; self.confidence_level = 0.95; self.perturbation_range = 0.1; self.params_to_perturb_names = []


        logger.info("Initialized RobustnessAnalyzer.")
        logger.info(f"Base Strategy Params (subset): {list(self.base_strategy_params.keys())}")
        logger.info(f"Parameters selected for perturbation: {self.params_to_perturb_names}")


    def monte_carlo_simulation(self) -> Dict:
        """
        Perform Monte Carlo simulation by perturbing strategy parameters.
        Returns analysis results including statistics, confidence intervals, stability scores, and raw metrics.
        """
        logger.info("="*40)
        logger.info(" DÉBUT DE L'ANALYSE DE ROBUSTESSE (MONTE CARLO) ")
        logger.info(f"Nombre de simulations: {self.n_simulations}")
        logger.info(f"Intervalle de perturbation: +/- {self.perturbation_range*100:.2f}%")
        logger.info(f"Niveau de confiance pour C.I.: {self.confidence_level*100:.0f}%")


        results = []

        for sim_num in tqdm(range(self.n_simulations), desc="Running MC Simulations"):
            try:
                # 1. Generate perturbed parameters
                perturbed_params = self._perturb_parameters()

                # 2. Instantiate and run Backtest with perturbed parameters
                # Pass the perturbed params to the Backtester instance
                sim_backtester = self.backtester_class(
                    prepared_data=self.prepared_data,
                    config=self.config, # Pass original config for backtest params
                    loaded_models=self.loaded_models,
                    strategy_params=perturbed_params # Pass the perturbed params dict
                )

                # Run the backtest. It returns a dict including 'metrics'.
                sim_result = sim_backtester.run_backtest()

                # 3. Extract key metrics from the backtest result['metrics']
                metrics = sim_result.get('metrics')

                # Check if essential metrics are valid
                if metrics and \
                   np.isfinite(metrics.get('total_return_percent', np.nan)) and \
                   np.isfinite(metrics.get('sharpe_ratio', np.nan)) and \
                   np.isfinite(metrics.get('max_drawdown_percent', np.nan)) and \
                   np.isfinite(metrics.get('win_rate', np.nan)) and \
                   np.isfinite(metrics.get('profit_factor', np.nan)):

                   metrics_to_keep = {
                       'total_return_percent': metrics['total_return_percent'],
                       'sharpe_ratio': metrics['sharpe_ratio'],
                       'max_drawdown_percent': metrics['max_drawdown_percent'],
                       'win_rate': metrics['win_rate'],
                       'profit_factor': metrics['profit_factor'],
                       # Add other numeric metrics if desired and available
                   }
                   results.append(metrics_to_keep)

                else:
                    logger.warning(f"Simulation {sim_num + 1} resulted in invalid or missing metrics. Skipping this run.")


            except Exception as e:
                logger.error(f"Simulation {sim_num + 1} failed with error: {e}", exc_info=True)
                continue

        # --- Analysis of Results ---
        if not results:
            raise ValueError("All simulations failed or produced invalid results. Please check your backtesting setup and data.")

        return self._analyze_simulation_results(results)


    def _perturb_parameters(self) -> Dict:
        """
        Creates a dictionary of perturbed strategy parameters based on the base parameters.
        Perturbs parameters listed in self.params_to_perturb_names from the base_strategy_params dict.
        """
        perturbed = self.base_strategy_params.copy() # Start with a copy of the base parameters
        perturbation_range = self.perturbation_range

        # logger.debug("Perturbing strategy parameters...") # Too verbose

        for param_name_lower in self.params_to_perturb_names:
             # Find the original key name (case-sensitive) in base_strategy_params
             original_param_name = None
             for key in self.base_strategy_params.keys():
                  if key.lower() == param_name_lower:
                       original_param_name = key
                       break

             if original_param_name and original_param_name in perturbed:
                base_value_str = str(perturbed[original_param_name])
                try:
                    # Attempt to infer type or use float by default
                    if base_value_str.lower() in ('true', 'false'):
                         base_value = bool(base_value_str) # Not perturbing bools usually
                         continue # Skip perturbing booleans
                    elif '.' in base_value_str or 'e' in base_value_str.lower():
                         base_value_numeric = float(base_value_str)
                         value_type = float
                    else:
                         base_value_numeric = int(base_value_str)
                         value_type = int # Infer int


                except ValueError:
                    logger.warning(f"Parameter '{original_param_name}' has non-numeric base value '{base_value_str}'. Skipping perturbation.")
                    continue

                # Generate random factor
                perturb_factor = random.uniform(1 - perturbation_range, 1 + perturbation_range)

                # Apply perturbation
                perturbed_value_numeric = base_value_numeric * perturb_factor

                # Apply bounds based on parameter name heuristic
                if 'threshold' in param_name_lower or 'ratio' in param_name_lower or 'percent' in param_name_lower or 'multiplier' in param_name_lower:
                     # Common bounds: > 0
                     perturbed_value_numeric = max(1e-9, perturbed_value_numeric) # Ensure positive (use 1e-9 to avoid division by 0)

                # Probabilities thresholds <= 1.0
                if 'prob_threshold' in param_name_lower:
                     perturbed_value_numeric = min(1.0, perturbed_value_numeric)

                # For integer parameters, round the perturbed value
                if value_type == int:
                     perturbed_value = int(max(1, round(perturbed_value_numeric))) # Ensure minimum 1 for periods/counts
                else:
                     perturbed_value = perturbed_value_numeric # Keep as float


                perturbed[original_param_name] = perturbed_value # Store with original case key
                # logger.debug(f"Param '{original_param_name}': Base={base_value_str}, Perturbed={perturbed_value_numeric:.4f} (Final: {perturbed_value})") # Too verbose

            # else: logger.warning(f"Parameter '{param_name_lower}' listed for perturbation not found in base_strategy_params.") # Too verbose

        return perturbed


    def _analyze_simulation_results(self, results: List[Dict]) -> Dict:
        """
        Analyzes the collected simulation metrics.
        Args:
            results: List of dictionaries, each containing metrics for one simulation.
        Returns:
            Dictionary with statistical summary, confidence intervals, and stability scores.
        """
        logger.info("\nAnalyzing simulation results...")
        df_results = pd.DataFrame(results)

        if df_results.empty:
            logger.error("DataFrame de résultats de simulation est vide après filtrage.")
            return {}

        # Basic statistics (mean, std, etc.)
        stats = df_results.describe().T.to_dict('index') # Transpose .describe() for better readability

        # Calculate confidence intervals (using percentile method)
        confidence_intervals = {}
        for metric in df_results.columns:
            try:
                lower = np.percentile(df_results[metric], (1 - self.confidence_level) * 100 / 2)
                upper = np.percentile(df_results[metric], (1 + self.confidence_level) * 100 / 2)
                confidence_intervals[metric] = (float(lower), float(upper))
            except Exception as e:
                logger.warning(f"Could not calculate percentile CI for metric '{metric}': {e}. Setting to NaN.")
                confidence_intervals[metric] = (np.nan, np.nan)

        # Calculate stability scores (coefficient of variation = std / |mean|). Lower is better.
        stability_scores = {}
        for metric in df_results.columns:
             mean_val = df_results[metric].mean()
             std_val = df_results[metric].std()
             if abs(mean_val) > 1e-9 and not np.isnan(std_val):
                 stability_scores[metric] = float(std_val / abs(mean_val))
             else:
                 stability_scores[metric] = np.nan

        logger.info("\n=== Robustness Analysis Results ===")
        logger.info(f"Number of successful simulations analyzed: {len(df_results)}")

        logger.info("\nMean Performance Metrics:")
        for metric, summary in stats.items():
             if 'mean' in summary:
                 logger.info(f"  {metric}: {summary['mean']:.4f}")

        logger.info("\nStability Scores (lower is better, NaN if mean near zero):")
        for metric, score in stability_scores.items():
            logger.info(f"  {metric}: {score:.4f}")

        logger.info(f"\nConfidence Intervals ({self.confidence_level*100:.0f}%):")
        for metric, (lower, upper) in confidence_intervals.items():
            logger.info(f"  {metric}: [{lower:.4f}, {upper:.4f}]")

        # Visualization
        self._plot_simulation_results(df_results, confidence_intervals, self.confidence_level)

        return {
            'statistics': stats,
            'confidence_intervals': confidence_intervals,
            'stability_scores': stability_scores,
            'raw_results': df_results
        }

    def _plot_simulation_results(self, df_results: pd.DataFrame, confidence_intervals: Dict, confidence_level: float):
        """Create visualizations of simulation results."""
        logger.info("\nGenerating visualization plots...")

        metrics_to_plot = [m for m in ['total_return_percent', 'sharpe_ratio', 'max_drawdown_percent', 'win_rate', 'profit_factor'] if m in df_results.columns]
        if not metrics_to_plot:
             logger.warning("No metrics available to plot.")
             return

        n_metrics = len(metrics_to_plot)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 5))

        # Handle case where axes is not a 2D array (1x1, 1xN, Nx1)
        if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
        elif n_rows == 1: axes = np.expand_dims(axes, axis=0)
        elif n_cols == 1: axes = np.expand_dims(axes, axis=1)

        axes = axes.flatten()

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            sns.histplot(df_results[metric], kde=True, ax=ax, bins=30)

            if metric in confidence_intervals:
                lower, upper = confidence_intervals[metric]
                if np.isfinite(lower):
                    ax.axvline(lower, color='r', linestyle='--', label=f'{(1-confidence_level)*100/2:.1f}% CI')
                if np.isfinite(upper):
                     ax.axvline(upper, color='r', linestyle='--')
                mean_value = df_results[metric].mean()
                if np.isfinite(mean_value):
                    ax.axvline(mean_value, color='g', linestyle='-', label='Mean')

            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel(metric)
            ax.set_ylabel('Count')
            ax.legend()

        for j in range(i + 1, len(axes)): fig.delaxes(axes[j])

        plt.suptitle('Monte Carlo Simulation Results', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()


# --- Public Interface Functions for Notebook ---

def _load_models(config: ConfigParser) -> Dict:
    """
    Charge les modèles ML, le scaler et les paramètres d'entraînement.
    
    Args:
        config: Configuration
        
    Returns:
        Dict avec les modèles chargés
    """
    # Initialiser les modules pour charger depuis les fichiers
    feature_engineer.initialize_feature_scalers(config)
    model_xgboost.initialize(config)
    model_futurequant.initialize(config)
    
    # Charger les paramètres d'entraînement globaux (noms des features, quantiles)
    training_params_file = config.get('training', 'training_params_file', fallback='training_params.json')
    loaded_training_params = {}
    
    try:
        if os.path.exists(training_params_file):
            with open(training_params_file, 'r') as f:
                loaded_training_params = json.load(f)
            logger.info(f"Paramètres d'entraînement chargés depuis {training_params_file}.")
        else:
            logger.warning(f"Fichier de paramètres '{training_params_file}' non trouvé. Les modèles pourraient ne pas être complètement configurés.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des paramètres d'entraînement: {e}")
    
    # Préparer le dictionnaire des modèles et paramètres
    loaded_models = {
        'xgb': model_xgboost.get_model(),
        'fq': model_futurequant.get_model(),
        'scaler': feature_engineer.get_scaler('feature_scaler'),
        'params': loaded_training_params
    }
    
    return loaded_models

def load_all_components_for_analysis(config_path: str = CONFIG_FILE) -> Dict:
    """
    Charge tous les composants nécessaires pour l'analyse de backtest.
    
    Args:
        config_path: Chemin du fichier de configuration
    
    Returns:
        Dict contenant 'config', 'prepared_data', 'loaded_models'
    """
    global _prepared_data_global  # Pour rendre prepared_data accessible aux fonctions de plotting
    
    # Charger la configuration
    logger.info(f"Chargement de la configuration depuis {config_path}")
    config = ConfigParser()
    config.read(config_path)
    
    # Initialiser les modules nécessaires avec la configuration
    logger.info("Initialisation des modules avec la configuration...")
    indicators.initialize(config)  # Initialiser les indicators (Pi-ratings)
    market_structure_analyzer.initialize(config)  # Initialiser l'analyseur de structure de marché
    order_flow_analyzer.initialize(config)  # Initialiser l'analyseur de flux d'ordres
    
    # Charger les données préparées (par exemple, depuis training_pipeline)
    logger.info("Chargement des données préparées via training_pipeline...")
    try:
        # Utiliser training_pipeline pour charger les données
        prepared_data = training_pipeline.load_prepared_data_for_analysis(config)
        _prepared_data_global = prepared_data  # Stocker dans la variable globale pour plotting
        
        logger.info(f"Données chargées: {len(prepared_data)} actifs")
        for symbol, df in prepared_data.items():
            logger.info(f"  - {symbol}: {len(df)} barres, {df.index.min()} à {df.index.max()}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        logger.error(traceback.format_exc())
        prepared_data = {}
    
    # Charger les modèles (scaler, XGBoost, FutureQuant etc.)
    logger.info("Chargement des modèles...")
    try:
        loaded_models = _load_models(config)
        logger.info(f"Modèles chargés: {loaded_models.keys()}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modèles: {e}")
        logger.error(traceback.format_exc())
        loaded_models = {}
    
    logger.info("Tous les composants ont été chargés avec succès.")

    return {
        'config': config,
        'prepared_data': prepared_data,
        'loaded_models': loaded_models
    }

def execute_backtest(components: Dict, strategy_params_override: Optional[Dict] = None) -> Dict:
    """
    Executes a single backtest run.
    Args:
        components: Dictionary returned by load_all_components_for_analysis.
        strategy_params_override: Optional dictionary of strategy parameters to use instead of config [strategy_hybrid].
                                 Used by robustness analysis.
    Returns:
        The result dictionary from Backtester.run_backtest().
    """
    logger.info("Executing backtest...")

    config = components.get('config')
    prepared_data = components.get('prepared_data')
    loaded_models = components.get('loaded_models')

    if config is None or prepared_data is None or loaded_models is None:
        raise ValueError("Invalid components dictionary provided. Missing config, prepared_data, or loaded_models.")

    # Get base strategy parameters from config if no override is provided
    if strategy_params_override is None:
        logger.info("No strategy_params_override provided. Loading base strategy parameters from config [strategy_hybrid].")
        try:
            # Get raw string values from config section items
            base_strategy_params = dict(config.items('strategy_hybrid'))
        except NoSectionError:
            logger.error("[strategy_hybrid] section missing in config. Cannot run backtest.")
            raise ValueError("Strategy parameters missing.")
        except Exception as e:
             logger.error(f"Error loading base strategy parameters from config: {e}", exc_info=True)
             raise ValueError("Error loading strategy parameters.")
    else:
        # Use the provided override parameters (already in the correct format, strings/floats depending on source)
        base_strategy_params = strategy_params_override
        logger.debug("Using strategy_params_override for backtest.")


    # Instantiate and run the Backtester
    try:
        backtester_instance = Backtester(
            prepared_data=prepared_data,
            config=config, # Pass original config for backtest/trading params
            loaded_models=loaded_models,
            strategy_params=base_strategy_params # Pass the strategy parameters dict
        )
        backtest_result = backtester_instance.run_backtest()
        return backtest_result

    except Exception as e:
        logger.error(f"Error during backtest execution: {e}", exc_info=True)
        raise


def execute_robustness_analysis(components: Dict) -> Dict:
    """
    Executes the Monte Carlo robustness analysis.
    Args:
        components: Dictionary returned by load_all_components_for_analysis.
    Returns:
        The result dictionary from RobustnessAnalyzer.monte_carlo_simulation().
    """
    logger.info("Executing robustness analysis...")

    config = components.get('config')
    prepared_data = components.get('prepared_data')
    loaded_models = components.get('loaded_models')

    if config is None or prepared_data is None or loaded_models is None:
        raise ValueError("Invalid components dictionary provided. Missing config, prepared_data, or loaded_models.")

    # Get base strategy parameters from config [strategy_hybrid]
    try:
        base_strategy_params = dict(config.items('strategy_hybrid'))
    except NoSectionError:
        logger.error("[strategy_hybrid] section missing in config. Cannot run robustness analysis.")
        raise ValueError("Strategy parameters missing for robustness analysis.")
    except Exception as e:
         logger.error(f"Error loading base strategy parameters for robustness analysis: {e}", exc_info=True)
         raise ValueError("Error loading strategy parameters.")


    # Instantiate and run the RobustnessAnalyzer
    try:
        robustness_analyzer_instance = RobustnessAnalyzer(
            backtester_class=Backtester, # Pass the Backtester class reference
            base_strategy_params=base_strategy_params, # Pass the base strategy parameters dict
            prepared_data=prepared_data,
            config=config, # Pass config for robustness params AND backtest params
            loaded_models=loaded_models # Pass loaded models/scaler
        )
        robustness_analysis_results = robustness_analyzer_instance.monte_carlo_simulation()
        return robustness_analysis_results

    except Exception as e:
        logger.error(f"Error during robustness analysis execution: {e}", exc_info=True)
        raise


# --- Visualization Functions (from example code, adapted) ---

def plot_equity_vs_asset(backtest_result: Dict, asset_key: Optional[str] = None, benchmark_wallet_series: Optional[pd.Series] = None, title: str = "Équité vs Actif", figsize: Tuple[int, int] = (12, 8)):
    """
    Trace la courbe d'équité de la stratégie par rapport au prix de l'actif principal.
    Args:
        backtest_result: Result dictionary from Backtester.run_backtest().
                         Must contain 'days' DataFrame with 'wallet' and DatetimeIndex.
        asset_key: Optional symbol@market key of the asset whose price to plot (e.g., 'BTC/USD@ccxt').
                   If None, the first asset in prepared_data is used (requires access to original prepared_data).
                   Or, ideally, prepared_data included in components or passed here.
        benchmark_wallet_series: Optional pandas Series of benchmark equity (DatetimeIndex, wallet).
        title: Plot title.
        figsize: Figure size.
    Returns:
        The matplotlib figure.
    """
    if backtest_result is None or 'days' not in backtest_result or backtest_result['days'].empty:
        logger.error("Backtest results missing or empty 'days' DataFrame. Cannot plot equity.")
        return None

    days_df = backtest_result['days'].copy()
    if not isinstance(days_df.index, pd.DatetimeIndex):
        logger.error("Days DataFrame index is not DatetimeIndex. Cannot plot.")
        return None
    if not pd.api.types.is_numeric_dtype(days_df['wallet']):
         logger.error("La colonne 'wallet' dans days_df n'est pas numérique. Impossible de calculer les retours mensuels.")
         return None

    # Scale equity to start at 100 for relative comparison if desired, or use raw wallet
    # Let's plot raw wallet for direct value tracking.
    days_df['wallet'].plot(ax=ax, label='Stratégie', color='blue')

    # Find the main asset data to plot its price
    # This function doesn't have direct access to the full prepared_data.
    # The prepared_data should be passed in the backtest_result or accessed globally (not ideal).
    # Let's assume the prepared_data is accessible via a global variable for simplicity in this notebook context.
    # In a production module, prepared_data should be passed.

    # --- Assuming prepared_data is accessible globally (for notebook context) ---
    global _prepared_data_global # Placeholder for global access in this sketch

    asset_price_series = None
    if asset_key:
        if _prepared_data_global and asset_key in _prepared_data_global:
             asset_price_df = _prepared_data_global[asset_key]
             if not asset_price_df.empty and 'close' in asset_price_df.columns and isinstance(asset_price_df.index, pd.DatetimeIndex):
                 # Align price series to the days_df index
                 asset_price_series = asset_price_df['close'].reindex(days_df.index, method='nearest') # Or ffill/bfill
                 logger.debug(f"Price series for {asset_key} found and aligned for plotting.")
             else:
                 logger.warning(f"Prepared data for {asset_key} missing, empty, or missing 'close' column/DatetimeIndex. Cannot plot price.")
        else:
             logger.warning(f"Prepared data for asset key '{asset_key}' not found globally. Cannot plot price.")

    elif _prepared_data_global and len(_prepared_data_global) > 0:
         # Use the first asset's price if asset_key is not specified
         first_asset_key = list(_prepared_data_global.keys())[0]
         first_asset_df = _prepared_data_global[first_asset_key]
         if not first_asset_df.empty and 'close' in first_asset_df.columns and isinstance(first_asset_df.index, pd.DatetimeIndex):
              asset_price_series = first_asset_df['close'].reindex(days_df.index, method='nearest')
              logger.warning(f"No asset_key specified. Using price of first prepared asset: {first_asset_key}.")
         else:
              logger.warning("First prepared asset data missing, empty, or missing 'close' column/DatetimeIndex. Cannot plot price.")


    # Plot asset price on a secondary axis
    if asset_price_series is not None and not asset_price_series.empty:
        ax2 = ax.twinx()
        # Scale asset price to match initial equity for visual comparison scale
        initial_equity = days_df['wallet'].iloc[0]
        initial_price = asset_price_series.iloc[0] if not asset_price_series.empty else 1.0
        scaled_price_series = (asset_price_series / initial_price) * initial_equity if initial_price > 0 else asset_price_series

        scaled_price_series.plot(ax=ax2, label=f'Prix {asset_key if asset_key else ""}', color='gray', alpha=0.5)
        ax2.set_ylabel('Prix Scalé')
        ax2.legend(loc='upper right')


    # Plot benchmark if provided (assuming it's a wallet series with DatetimeIndex)
    if benchmark_wallet_series is not None:
        if not isinstance(benchmark_wallet_series.index, pd.DatetimeIndex):
             logger.warning("Benchmark wallet series index is not DatetimeIndex. Cannot plot benchmark.")
        else:
             # Scale benchmark to match initial strategy equity
             initial_benchmark_equity = benchmark_wallet_series.iloc[0]
             scaled_benchmark_series = (benchmark_wallet_series / initial_benchmark_equity) * initial_equity if initial_benchmark_equity > 0 else benchmark_wallet_series

             scaled_benchmark_series.plot(ax=ax, label='Benchmark', color='green')


    ax.set_title(title)
    ax.set_ylabel('Équité')
    ax.legend(loc='upper left')

    plt.grid(True)
    plt.tight_layout()
    plt.show() # Display the plot in the notebook

    return fig


def plot_future_simulations_monte_carlo(
    df_historical_price: pd.DataFrame, # Historical data with 'close' and DatetimeIndex for baseline
    n_simulations: int = 100,
    days_forward: int = 30,
    volatility_factor: float = 1.0,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Trace les simulations Monte Carlo des mouvements futurs des prix.
    Args:
        df_historical_price: DataFrame with historical price data ('close' column, DatetimeIndex).
        n_simulations: Number of simulations.
        days_forward: Number of days to simulate forward.
        volatility_factor: Factor to adjust historical volatility (sigma).
        figsize: Figure size.
    Returns:
        The matplotlib figure.
    """
    if df_historical_price is None or df_historical_price.empty or 'close' not in df_historical_price.columns or not isinstance(df_historical_price.index, pd.DatetimeIndex):
        logger.error("Historical price data missing, empty, or missing 'close' column/DatetimeIndex for future simulation.")
        return None

    # Ensure historical data is daily for simplicity in simulating 'days_forward'
    # If timeframe is different, need to adjust mu and sigma calculation and date_range
    # Let's assume input df_historical_price is daily data.
    df_daily_history = df_historical_price.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna() # Aggregate to daily


    last_price = df_daily_history['close'].iloc[-1]
    returns = df_daily_history['close'].pct_change().dropna()

    if returns.empty:
         logger.warning("Not enough historical data to calculate returns for future simulation.")
         # Maybe just plot the last price and return?
         fig, ax = plt.subplots(figsize=figsize)
         ax.plot(df_daily_history.index, df_daily_history['close'], label='Historique', color='black')
         ax.scatter([df_daily_history.index[-1]], [last_price], color='red', zorder=5) # Mark last point
         ax.set_title("Simulation future impossible (pas assez de données historiques).")
         ax.legend(); ax.grid(True); plt.tight_layout(); plt.show(); return fig


    # Use geometric Brownian motion model assumptions for simulation
    mu = returns.mean() # Mean daily return
    sigma = returns.std() * volatility_factor # Adjusted daily volatility

    # Simule price paths
    # Start from the last price of historical data
    simulations = np.zeros((days_forward, n_simulations))
    simulations[0, :] = last_price # First row is the starting price for all simulations

    # Simulate each day's return and calculate the price
    daily_returns_sim = np.random.normal(mu, sigma, (days_forward -1, n_simulations))
    price_paths = np.cumprod(np.exp(daily_returns_sim), axis=0) * last_price

    # Combine the starting price row
    simulations[1:, :] = price_paths # Assign paths starting from day 2 (index 1)

    # Create future dates starting from the day *after* the last historical date
    last_date = df_daily_history.index[-1]
    future_dates = pd.date_range(start=last_date, periods=days_forward + 1, freq='D')[1:] # Daily frequency


    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot historical close prices
    df_daily_history['close'].plot(ax=ax, label='Historique', color='black', linewidth=2)

    # Plot simulated paths
    for i in range(n_simulations):
        ax.plot(future_dates, simulations[:, i], alpha=0.1, color='blue')

    # Plot mean simulation path
    mean_simulation = simulations.mean(axis=1)
    ax.plot(future_dates, mean_simulation, color='red', linewidth=2, label='Simulation Moyenne')

    # Plot percentiles paths
    percentiles = np.percentile(simulations, [10, 50, 90], axis=1) # Add 50th percentile (median)
    ax.plot(future_dates, percentiles[0], color='orange', linestyle='--', label='10ème Percentile')
    ax.plot(future_dates, percentiles[1], color='purple', linestyle='--', label='50ème Percentile (Médiane)')
    ax.plot(future_dates, percentiles[2], color='green', linestyle='--', label='90ème Percentile')


    ax.set_title(f'Simulation Monte Carlo ({n_simulations} exécutions, {days_forward} jours)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Prix')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return fig


def plot_backtest_monthly_returns(backtest_result: Dict, figsize: Tuple[int, int] = (14, 8)):
    """
    Trace un graphique à barres des retours mensuels du backtest.
    Args:
        backtest_result: Result dictionary from Backtester.run_backtest().
                         Must contain 'days' DataFrame with 'wallet' and DatetimeIndex.
        figsize: Figure size.
    Returns:
        The matplotlib figure.
    """
    if backtest_result is None or 'days' not in backtest_result or backtest_result['days'].empty:
        logger.error("Backtest results missing or empty 'days' DataFrame. Cannot plot monthly returns.")
        return None

    days_df = backtest_result['days'].copy()
    if not isinstance(days_df.index, pd.DatetimeIndex):
        logger.error("Days DataFrame index is not DatetimeIndex. Cannot plot monthly returns.")
        return None
    if not pd.api.types.is_numeric_dtype(days_df['wallet']):
         logger.error("La colonne 'wallet' dans days_df n'est pas numérique. Impossible de calculer les retours mensuels.")
         return None

    # Calculate daily returns if not already present and index is DatetimeIndex
    if 'daily_return' not in days_df.columns or not pd.api.types.is_numeric_dtype(days_df['daily_return']):
         days_df['daily_return'] = days_df['wallet'].pct_change().fillna(0)

    # Calculate monthly returns
    monthly_returns = days_df['daily_return'].resample('M').apply(lambda x: (1 + x).prod() - 1) # Cumulative return per month


    if monthly_returns.empty:
        logger.warning("Aucun retour mensuel calculé.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Aucun retour mensuel à afficher")
        plt.show(); return fig


    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars
    colors = ['green' if x > 0 else 'red' for x in monthly_returns]
    monthly_returns.plot(kind='bar', ax=ax, color=colors)

    # Add value labels above bars
    for i, v in enumerate(monthly_returns):
        text_val = f'{v:.2%}' if abs(v) < 0.1 else f'{v:.1%}' # Format as percentage
        ax.text(i, v + (0.005 * max(abs(monthly_returns.max()), abs(monthly_returns.min())) if v >= 0 else -0.005 * max(abs(monthly_returns.max()), abs(monthly_returns.min()))),
                text_val,
                ha='center', va='bottom' if v >= 0 else 'top',
                fontsize=8 if len(monthly_returns) > 24 else 9) # Adjust font size based on number of bars


    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_title('Retours Mensuels (%)')
    ax.set_xlabel('Mois')
    ax.set_ylabel('Retour (%)')

    # Format x-axis labels to show month and year
    ax.set_xticklabels([d.strftime('%b %Y') for d in monthly_returns.index], rotation=45, ha='right')

    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return fig


# --- Main Execution for Script/Notebook Usage Example ---
if __name__ == "__main__":
    # Example Usage in a Notebook Cell:
    #
    # import trading_analyzer as ta_bot
    # from configparser import ConfigParser
    #
    # # Optional: Configure logging for the notebook session
    # ta_bot.setup_logging(logging.INFO)
    #
    # # 1. Load all components (config, prepared data, models, scaler, params)
    # try:
    #     components = ta_bot.load_all_components_for_analysis(config_path='config.ini')
    # except Exception as e:
    #     print(f"Failed to load components: {e}")
    #     # Handle error, maybe exit cell or notebook
    #     components = None # Ensure components is None if loading failed
    #
    # if components:
    #     # Access components
    #     config = components['config']
    #     prepared_data = components['prepared_data'] # This is the dict {symbol_market: df_with_all_features}
    #     loaded_models = components['loaded_models'] # Dict with xgb, fq, scaler, params
    #
    #     # Make prepared_data accessible globally for plotting helpers (if not passing it)
    #     ta_bot._prepared_data_global = prepared_data
    #
    #     # Get the base strategy parameters from the loaded config
    #     try:
    #         base_strategy_params_from_config = dict(config.items('strategy_hybrid'))
    #     except:
    #         base_strategy_params_from_config = {} # Empty dict if section missing
    #
    #
    #     # 2. Execute the base backtest (using parameters from config.ini)
    #     print("\nRunning Base Backtest...")
    #     try:
    #         base_backtest_result = ta_bot.execute_backtest(components=components, strategy_params_override=base_strategy_params_from_config)
    #         # base_backtest_result contains 'wallet_final_cash', 'final_portfolio_value', 'trades', 'days', 'metrics'
    #
    #         # 3. Plot Base Backtest Results
    #         print("\nPlotting Base Backtest Results...")
    #         # Plot equity vs asset price (need to specify an asset key from your config/data)
    #         # Assuming 'BTC/USD@ccxt' is one of your assets in prepared_data
    #         if prepared_data and 'BTC/USD@ccxt' in prepared_data:
    #             ta_bot.plot_equity_vs_asset(base_backtest_result, asset_key='BTC/USD@ccxt', title='Backtest Stratégie vs BTC/USD')
    #         else:
    #              ta_bot.plot_equity_vs_asset(base_backtest_result, title='Backtest Stratégie') # Plot without asset price if not available
    #
    #         # Plot monthly returns
    #         ta_bot.plot_backtest_monthly_returns(base_backtest_result)
    #
    #         # Print detailed metrics
    #         print("\nBase Backtest Metrics:")
    #         for metric, value in base_backtest_result['metrics'].items():
    #              print(f"  {metric}: {value}")
    #
    #
    #     except Exception as e:
    #         print(f"Error during base backtest or plotting: {e}")
    #
    #
    #     # 4. Execute Robustness Analysis (Monte Carlo)
    #     print("\nRunning Robustness Analysis...")
    #     try:
    #         robustness_analysis_results = ta_bot.execute_robustness_analysis(components=components)
    #         # robustness_analysis_results contains 'statistics', 'confidence_intervals', 'stability_scores', 'raw_results'
    #         # Plots are generated internally by RobustnessAnalyzer.monte_carlo_simulation()
    #
    #         print("\nRobustness Analysis Complete.")
    #         # You can access and print detailed results from robustness_analysis_results if needed
    #
    #     except Exception as e:
    #         print(f"Error during robustness analysis: {e}")
    #
    #
    #     # 5. Example of Future Simulation (Optional)
    #     print("\nRunning Future Price Simulation (Example)...")
    #     try:
    #         # Requires historical price data for *one* asset to base the simulation on.
    #         # Use the same asset key as for equity plot.
    #         sim_asset_key = 'BTC/USD@ccxt' # Choose an asset from your prepared_data
    #         if prepared_data and sim_asset_key in prepared_data:
    #              # Get just the historical close price data for this asset (need the full history used for training)
    #              # This data is available in prepared_data but might be filtered/modified.
    #              # Need to access the *original* historical data used by the pipeline before cleaning/filtering for ML if possible,
    #              # or pass the relevant slice here.
    #              # Let's simplify: use the close price from the prepared_data DataFrame for this asset, assuming it has enough history.
    #              sim_historical_df = prepared_data[sim_asset_key] # This dataframe already has the necessary history
    #              if not sim_historical_df.empty and 'close' in sim_historical_df.columns:
    #                 ta_bot.plot_future_simulations_monte_carlo(
    #                     df_historical_price=sim_historical_df[['close']], # Pass only close price column
    #                     n_simulations=200, # Fewer simulations for faster example
    #                     days_forward=60,
    #                     volatility_factor=1.0
    #                 )
    #              else:
    #                  print(f"Prepared data for {sim_asset_key} missing 'close' or empty for future simulation.")
    #
    #         else:
    #              print(f"Prepared data for future simulation asset '{sim_asset_key}' not found.")
    #
    #     except Exception as e:
    #         print(f"Error during future simulation: {e}")
    #
    #
    #     print("\nAnalysis session finished.")
    # else:
        print("\nAnalysis session cancelled due to failure in loading components.")