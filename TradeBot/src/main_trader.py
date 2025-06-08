import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# main_trader.py
import logging
import time
import json
import numpy as np
import pandas as pd
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from configparser import ConfigParser, NoSectionError, NoOptionError
import ccxt
import traceback
import smtplib # For notifications
from email.mime.text import MIMEText # For notifications
import requests # For Discord notifications

# --- Import project modules from src ---
# Assuming src is in PYTHONPATH or main_trader.py is run from project root
from src import indicators # Expected to contain IndicatorsManager class
from src.pipeline import feature_engineer # Expected to contain FeatureEngineer class
from src.models import model_xgboost, model_futurequant # Expected to provide model loading/prediction functions/classes
from src.strategies.strategy_hybrid import StrategyHybrid
from src.core import order_manager as core_order_manager # Alias to avoid potential naming conflicts
from src.core import position_manager as core_position_manager
from src.analysis.market_structure_analyzer import MarketStructureAnalyzer
from src.analysis.order_flow_analyzer import OrderFlowAnalyzer
from src.analysis.sentiment_analyzer import SentimentAnalyzer

# --- Constants ---
CONFIG_FILE_PATH = 'config.ini' # Adjusted to be relative to project root if main_trader.py is there
LOG_FILE_PATH = "logs/hybrid_trading_bot_improved.log" # Ensure 'logs' directory exists

# API Call Constants
MAX_API_RETRIES = 5
API_RETRY_DELAY_SECONDS = 5
API_BACKOFF_FACTOR = 2

# System Health & Monitoring Constants
HEARTBEAT_INTERVAL_SECONDS = 60
MAX_CONSECUTIVE_ERRORS_BEFORE_EMERGENCY = 5
MAX_ASSET_PROCESSING_TIMEOUT_SECONDS = 180 # Timeout for processing a single asset

# Default values from config if not found
DEFAULT_MAX_HISTORY_BARS = 500
DEFAULT_UPDATE_INTERVAL_SECONDS = 3600
DEFAULT_POSITION_SIZE_PERCENT = 0.02
DEFAULT_MAX_CONCURRENT_POSITIONS = 3
DEFAULT_TIMEZONE = "UTC"

# --- Setup Logging ---
logger = logging.getLogger("TradingBot")

# --- Core Helper/Manager Classes defined in main_trader.py ---
# These are either utility classes or complex managers tightly coupled with TradingBot's orchestration.
# For a larger project, some of these might also move to src.core.utils or similar.

class ConfigManager:
    def __init__(self, config_path: str = CONFIG_FILE_PATH):
        self.config = ConfigParser()
        if not os.path.exists(config_path):
            # Attempt to find it in a 'config' subdirectory if not at root
            alt_config_path = os.path.join('config', os.path.basename(config_path))
            if os.path.exists(alt_config_path):
                config_path = alt_config_path
            else:
                logger.error(f"Configuration file not found: {config_path} or {alt_config_path}")
                raise FileNotFoundError(f"Configuration file not found: {config_path} or {alt_config_path}")
        self.config.read(config_path)
        logger.info(f"Configuration loaded from {config_path}")

    def get(self, section: str, option: str, fallback: Any = None) -> str:
        try: return self.config.get(section, option)
        except (NoSectionError, NoOptionError): return fallback

    def getint(self, section: str, option: str, fallback: Any = None) -> int:
        try: return self.config.getint(section, option)
        except (NoSectionError, NoOptionError): return fallback

    def getfloat(self, section: str, option: str, fallback: Any = None) -> float:
        try: return self.config.getfloat(section, option)
        except (NoSectionError, NoOptionError): return fallback

    def getboolean(self, section: str, option: str, fallback: Any = None) -> bool:
        try: return self.config.getboolean(section, option)
        except (NoSectionError, NoOptionError): return fallback

    def has_section(self, section: str) -> bool:
        """Checks if the configuration has the specified section."""
        return self.config.has_section(section)

    def get_section(self, section: str) -> Dict[str, str]:
        """Returns a dictionary of options in a section."""
        try:
            return dict(self.config.items(section))
        except (NoSectionError, NoOptionError):
            return {}


class ModelManager:
    def __init__(self, config_manager: ConfigManager, feature_engineer_instance: feature_engineer.FeatureEngineer):
        self.config = config_manager
        self.feature_engineer = feature_engineer_instance # Instance of imported FeatureEngineer
        self.xgb_model: Optional[Any] = None # Will hold the loaded XGBoost model object
        self.fq_model: Optional[Any] = None  # Will hold the loaded FutureQuant model object
        self.training_params: Dict[str, Any] = {}
        self._load_models()
        self._load_training_params()
        logger.info("ModelManager initialized.")

    def _load_training_params(self):
        try:
            params_file = self.config.get('training', 'training_params_file', fallback='data/training_params.json')
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    self.training_params = json.load(f)
                logger.info(f"Training parameters loaded from {params_file}.")
            else:
                logger.warning(f"Training parameters file not found: {params_file}. Using empty params.")
                self.training_params = {}
        except Exception as e:
            logger.error(f"Error loading training_params.json: {e}")
            self.training_params = {}
            
    def _load_models(self):
        # XGBoost Model
        xgb_model_path = self.config.get('models', 'xgboost_path', fallback=None)
        if xgb_model_path and os.path.exists(xgb_model_path):
            try:
                # Assuming model_xgboost.load_model_from_path returns the actual model object
                self.xgb_model = model_xgboost.load_model_from_path(xgb_model_path)
                logger.info(f"XGBoost model loaded from {xgb_model_path} using 'src.models.model_xgboost'.")
            except Exception as e:
                logger.error(f"Failed to load XGBoost model from {xgb_model_path}: {e}", exc_info=True)
                self.xgb_model = None
        else:
            logger.warning(f"XGBoost model path '{xgb_model_path}' not found or not configured.")
            self.xgb_model = None

        # FutureQuant Model
        fq_model_path = self.config.get('models', 'futurequant_path', fallback=None)
        if fq_model_path and os.path.exists(fq_model_path):
            try:
                # Assuming model_futurequant.load_model_from_path returns the actual model object
                self.fq_model = model_futurequant.load_model_from_path(fq_model_path)
                logger.info(f"FutureQuant model loaded from {fq_model_path} using 'src.models.model_futurequant'.")
            except Exception as e:
                logger.error(f"Failed to load FutureQuant model from {fq_model_path}: {e}", exc_info=True)
                self.fq_model = None
        else:
            logger.warning(f"FutureQuant model path '{fq_model_path}' not found or not configured.")
            self.fq_model = None

    def get_xgb_model(self) -> Optional[Any]: return self.xgb_model
    def get_fq_model(self) -> Optional[Any]: return self.fq_model
    def get_feature_scaler(self) -> Optional[Any]: return self.feature_engineer.get_scaler()
    def get_xgb_feature_names(self) -> List[str]: return self.training_params.get('xgboost_features', [])
    def get_all_feature_names(self) -> List[str]: return self.training_params.get('feature_columns', [])

    def _prepare_features_for_model(self, features_series: pd.Series, model_feature_names: List[str]) -> Optional[pd.DataFrame]:
        if not model_feature_names:
            logger.error("Model feature names list is empty. Cannot prepare features.")
            return None
            
        model_input_dict = {}
        missing_features = []
        for feat_name in model_feature_names:
            if feat_name in features_series:
                model_input_dict[feat_name] = features_series[feat_name]
            else:
                model_input_dict[feat_name] = 0.0 
                missing_features.append(feat_name)
        
        if missing_features:
            logger.warning(f"Missing features for model input: {missing_features}. Filled with 0.0.")
            
        features_df = pd.DataFrame([model_input_dict], columns=model_feature_names)

        scaler = self.get_feature_scaler()
        if scaler:
            try:
                # Ensure columns are in the same order as when the scaler was fit
                # This assumes `model_feature_names` is the correct order.
                # If scaler was fit on a different order or subset, this needs adjustment.
                cols_to_scale = [col for col in model_feature_names if col in features_df.columns]
                if not cols_to_scale:
                    logger.warning("No common columns found between model_feature_names and features_df for scaling.")
                    return features_df
 
                scaled_values = scaler.transform(features_df[cols_to_scale].values)
                # Create a new DataFrame for scaled values, ensuring correct column names
                scaled_df_part = pd.DataFrame(scaled_values, columns=cols_to_scale, index=features_df.index)
                
                # Combine scaled columns with any non-scaled columns (if applicable)
                # This assumes all model_feature_names were intended to be scaled.
                # If some features are categorical or pre-scaled, this needs more complex logic.
                final_features_df = features_df.copy()
                final_features_df[cols_to_scale] = scaled_df_part
                
                return final_features_df
            except Exception as e:
                logger.error(f"Error scaling features for model: {e}. Using unscaled features.", exc_info=True)
                return features_df 
        else:
            logger.warning("No feature scaler available. Using unscaled features for model.")
            return features_df

    def predict_xgboost(self, latest_features_series: pd.Series) -> Optional[List[float]]:
        if not self.xgb_model: logger.debug("XGBoost model not loaded, cannot predict."); return None
        
        xgb_feature_names = self.get_xgb_feature_names()
        features_df_scaled = self._prepare_features_for_model(latest_features_series, xgb_feature_names)
        if features_df_scaled is None: return None
        
        try:
            predictions = self.xgb_model.predict_proba(features_df_scaled) 
            logger.debug(f"XGBoost raw prediction shape: {predictions.shape}")
            return predictions[0].tolist() 
        except Exception as e:
            logger.error(f"Error during XGBoost prediction: {e}", exc_info=True)
            return None


    def predict_futurequant(self, sequence_data: np.ndarray) -> Optional[List[float]]:
        if not self.fq_model: logger.debug("FutureQuant model not loaded, cannot predict."); return None
        try:
            prediction = self.fq_model.predict(sequence_data) 
            logger.debug(f"FutureQuant raw prediction shape: {prediction.shape}")
            return prediction[0].tolist() 
        except Exception as e:
            logger.error(f"Error during FutureQuant prediction: {e}", exc_info=True)
            return None


class NotificationService:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        logger.info("NotificationService initialized.")

    def send_email_notification(self, subject: str, body: str):
        if not self.config.getboolean('notifications', 'email_enabled', fallback=False): return
        try:
            smtp_server = self.config.get('notifications', 'smtp_server')
            smtp_port = self.config.getint('notifications', 'smtp_port') 
            smtp_user = self.config.get('notifications', 'smtp_user')
            smtp_password = self.config.get('notifications', 'smtp_password')
            from_email = self.config.get('notifications', 'from_email')
            to_email = self.config.get('notifications', 'to_email')

            if not all([smtp_server, smtp_user, smtp_password, from_email, to_email]):
                logger.warning("Email notification configuration is incomplete.")
                return

            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = from_email
            msg['To'] = to_email

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if self.config.getboolean('notifications', 'smtp_use_tls', fallback=True):
                    server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            logger.info(f"Email notification sent: {subject}")
        except (NoSectionError, NoOptionError) as e:
            logger.error(f"Email configuration error: {e}")
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}", exc_info=True)

    def send_discord_notification(self, message: str):
        if not self.config.getboolean('notifications', 'discord_enabled', fallback=False): return
        try:
            webhook_url = self.config.get('notifications', 'discord_webhook', fallback=None)
            if not webhook_url:
                logger.warning("Discord webhook URL not configured.")
                return
            payload = {"content": message, "username": "Trading Bot"}
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status() 
            logger.info("Discord notification sent.")
        except (NoSectionError, NoOptionError) as e:
            logger.error(f"Discord configuration error: {e}")
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}", exc_info=True)
            
    def send_emergency_notification(self, error_count: int, message: str = ""):
        subject = "🚨 CRITICAL ALERT - Trading Bot Emergency Shutdown 🚨"
        body = (f"The trading bot has triggered an emergency shutdown protocol.\n"
                f"Consecutive errors: {error_count}\n"
                f"Message: {message if message else 'Unspecified critical error.'}\n\n"
                f"Timestamp: {pd.Timestamp.now(tz=self.config.get('trading', 'timezone', fallback=DEFAULT_TIMEZONE)).isoformat()}\n"
                f"Please check the logs and system status immediately.")
        
        logger.critical(subject + "\n" + body)
        self.send_email_notification(subject, body)
        self.send_discord_notification(subject + "\n" + body)


class SystemStateManager:
    def __init__(self):
        self.consecutive_errors: int = 0
        self.last_heartbeat_time: float = time.time()
        self.emergency_shutdown_active: bool = False
        self.trading_lock = threading.Lock() 
        logger.info("SystemStateManager initialized.")

    def increment_consecutive_errors(self):
        with self.trading_lock: self.consecutive_errors += 1
    def reset_consecutive_errors(self):
        with self.trading_lock: self.consecutive_errors = 0
    def get_consecutive_errors(self) -> int:
        return self.consecutive_errors
    def is_emergency_shutdown(self) -> bool: return self.emergency_shutdown_active
    def trigger_emergency_shutdown(self):
        with self.trading_lock:
            if not self.emergency_shutdown_active:
                logger.critical("EMERGENCY SHUTDOWN TRIGGERED!")
                self.emergency_shutdown_active = True
    def update_heartbeat_time(self): self.last_heartbeat_time = time.time()


class HistoricalDataManager:
    def __init__(self, config_manager: ConfigManager, 
                 feature_engineer_instance: feature_engineer.FeatureEngineer, 
                 indicators_manager_instance: indicators.IndicatorsManager, 
                 max_history_length: int = 500):
        self.config = config_manager
        self.feature_engineer = feature_engineer_instance
        self.indicators_manager = indicators_manager_instance
        self.max_history_length = max_history_length
        self.historical_ohlcv: Dict[str, pd.DataFrame] = {} 
        self.ms_features_history: Dict[str, Dict[pd.Timestamp, Dict]] = {}
        self.of_features_history: Dict[str, Dict[pd.Timestamp, Dict]] = {}
        self.ta_features_history: Dict[str, pd.DataFrame] = {}
        self.sentiment_history: Dict[str, Dict[pd.Timestamp, float]] = {} 
        self.complete_features_history: Dict[str, pd.DataFrame] = {}
        self.market_regime_history: Dict[str, pd.Series] = {} # Added for market regime
        self.fq_window_size = config_manager.getint('futurequant', 'window_size_in', fallback=32)
        logger.info(f"HistoricalDataManager initialized with max_history={max_history_length}, FQ_window={self.fq_window_size}")

    def _init_asset_storage(self, symbol_market: str):
        if symbol_market not in self.historical_ohlcv:
            self.historical_ohlcv[symbol_market] = pd.DataFrame()
            self.ms_features_history[symbol_market] = {}
            self.of_features_history[symbol_market] = {}
            self.ta_features_history[symbol_market] = pd.DataFrame()
            self.sentiment_history[symbol_market] = {} 
            self.complete_features_history[symbol_market] = pd.DataFrame()
            self.market_regime_history[symbol_market] = pd.Series(dtype=str) # Added for market regime

    def add_ohlcv_data(self, symbol_market: str, df_ohlcv: pd.DataFrame):
        self._init_asset_storage(symbol_market)
        current_df = self.historical_ohlcv[symbol_market]
        if not df_ohlcv.empty:
            if not current_df.empty:
                if not isinstance(current_df.index, pd.DatetimeIndex): current_df.index = pd.to_datetime(current_df.index)
                if not isinstance(df_ohlcv.index, pd.DatetimeIndex): df_ohlcv.index = pd.to_datetime(df_ohlcv.index)
                
                logger.debug(f"Combining OHLCV for {symbol_market}. Current DF index TZ: {current_df.index.tz}, New DF index TZ: {df_ohlcv.index.tz}")
                combined_df = pd.concat([current_df, df_ohlcv]).reset_index().drop_duplicates(subset=['timestamp'], keep='last').set_index('timestamp')
            else:
                combined_df = df_ohlcv.copy()
            
            combined_df = combined_df.sort_index()
            self.historical_ohlcv[symbol_market] = combined_df.iloc[-self.max_history_length:]
            logger.debug(f"OHLCV data for {symbol_market} updated. Total bars: {len(self.historical_ohlcv[symbol_market])}")

            full_hist = self.historical_ohlcv[symbol_market]
            if not full_hist.empty:
                if self.indicators_manager and self.indicators_manager.is_initialized():
                    self.ta_features_history[symbol_market] = self.indicators_manager.calculate_all_ta_indicators(full_hist)
                    logger.debug(f"TA features for {symbol_market} updated. Shape: {self.ta_features_history[symbol_market].shape}")
                    
                    if not self.ta_features_history[symbol_market].empty and \
                       all(col in self.ta_features_history[symbol_market].columns for col in ['ADX', 'ATR', 'BB_WIDTH']):
                        try:
                            self.market_regime_history[symbol_market] = self.indicators_manager.classify_market_regime(
                                self.ta_features_history[symbol_market],
                                self.config_manager
                            )
                            logger.debug(f"Market regime for {symbol_market} updated. Shape: {self.market_regime_history[symbol_market].shape}")
                        except Exception as e:
                            logger.error(f"Error calculating market regime for {symbol_market}: {e}", exc_info=True)
                            self.market_regime_history[symbol_market] = pd.Series(dtype=str, index=full_hist.index).fillna("Indeterminate")
                    else:
                        logger.warning(f"Could not calculate market regime for {symbol_market} due to missing TA features (ADX, ATR, BB_WIDTH) or empty TA DataFrame.")
                        self.market_regime_history[symbol_market] = pd.Series(dtype=str, index=full_hist.index).fillna("Indeterminate")
                else:
                    logger.warning(f"IndicatorsManager not initialized for {symbol_market}, skipping TA and Regime calculation.")
                    self.market_regime_history[symbol_market] = pd.Series(dtype=str, index=full_hist.index).fillna("Indeterminate")


    def _add_timestamped_features(self, history_dict: Dict[str, Dict[pd.Timestamp, Dict]], symbol_market: str, timestamp: pd.Timestamp, features: Dict):
        self._init_asset_storage(symbol_market)
        history_dict[symbol_market][timestamp] = features.copy()
        if len(history_dict[symbol_market]) > self.max_history_length:
            sorted_timestamps = sorted(history_dict[symbol_market].keys())
            for ts_to_remove in sorted_timestamps[:-self.max_history_length]:
                del history_dict[symbol_market][ts_to_remove]

    def add_ms_features(self, symbol_market: str, timestamp: pd.Timestamp, ms_features: Dict): self._add_timestamped_features(self.ms_features_history, symbol_market, timestamp, ms_features)
    def add_of_features(self, symbol_market: str, timestamp: pd.Timestamp, of_features: Dict): self._add_timestamped_features(self.of_features_history, symbol_market, timestamp, of_features)

    def add_sentiment_score(self, symbol_market: str, timestamp: pd.Timestamp, score: float) -> None: 
        self._init_asset_storage(symbol_market)
        
        if pd.isna(timestamp):
            logger.warning(f"Timestamp invalide (NaN) pour le score de sentiment de {symbol_market}. Score non ajouté.")
            return
            
        self.sentiment_history[symbol_market][timestamp] = score
        
        timestamps = sorted(self.sentiment_history[symbol_market].keys())
        if len(timestamps) > self.max_history_length:
            for ts_old in timestamps[:-self.max_history_length]:
                if ts_old in self.sentiment_history[symbol_market]: 
                    del self.sentiment_history[symbol_market][ts_old]
        
        logger.debug(f"Score de sentiment {score:.4f} ajouté pour {symbol_market} à {timestamp}, "
                     f"historique: {len(self.sentiment_history[symbol_market])} entrées")

    def update_complete_features(self, symbol_market: str, current_processing_timestamp: Optional[pd.Timestamp] = None) -> None:
        self._init_asset_storage(symbol_market)
        df_ohlcv = self.historical_ohlcv.get(symbol_market)
        if df_ohlcv is None or df_ohlcv.empty:
            logger.warning(f"No OHLCV data for {symbol_market} to update complete features.")
            return

        ohlcv_to_process = df_ohlcv
        existing_complete_df = self.complete_features_history.get(symbol_market, pd.DataFrame())
        new_features_list = []

        if not existing_complete_df.empty:
            last_processed_ts = existing_complete_df.index.max()
            ohlcv_to_process = df_ohlcv[df_ohlcv.index > last_processed_ts]
        
        if current_processing_timestamp:
             ohlcv_to_process = ohlcv_to_process[ohlcv_to_process.index <= current_processing_timestamp]

        if ohlcv_to_process.empty:
            logger.debug(f"No new OHLCV data to build complete features for {symbol_market} up to {current_processing_timestamp or 'latest'}.")
            return

        for idx_ts, ohlcv_row_series in ohlcv_to_process.iterrows():
            ms_f = self.ms_features_history[symbol_market].get(idx_ts, {})
            of_f = self.of_features_history[symbol_market].get(idx_ts, {})
            pi_r = self.indicators_manager.get_ratings(symbol_market) 
            ta_f_row = self.ta_features_history[symbol_market].loc[idx_ts] if idx_ts in self.ta_features_history.get(symbol_market, pd.DataFrame()).index else pd.Series(dtype='float64')

            try:
                trinary_config_params = self.config.get_section('trinary_config') if self.config else {}
                complete_feat_series = self.feature_engineer.build_complete_raw_features(
                    ohlcv_row_series, ms_f, of_f, pi_r, ta_f_row, indicator_config=trinary_config_params
                )
                complete_feat_series.name = idx_ts 
                new_features_list.append(complete_feat_series)
            except Exception as e:
                logger.error(f"Error building complete features for {symbol_market} at {idx_ts}: {e}", exc_info=True)
        
        if not new_features_list:
            logger.debug(f"No new complete features generated for {symbol_market}.")
            return

        new_complete_features_df = pd.DataFrame(new_features_list)
        combined_df = pd.concat([existing_complete_df, new_complete_features_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()

        # === START BB_WIDTH Percentile Rank Calculation ===
        if 'BB_WIDTH' in combined_df.columns:
            bbw_percentile_window = self.config.getint('trinary_config', 'trinary_bbw_percentile_window', fallback=100)
            min_p = max(1, bbw_percentile_window // 2) 

            bb_width_pr = combined_df['BB_WIDTH'].rolling(
                window=bbw_percentile_window, 
                min_periods=min_p
            ).rank(pct=True)
            
            # Derive volatility_cycle_pos using the manager method, which maps [0,1] to [0, 2*PI]
            # and handles NaNs by defaulting to PI (median equivalent)
            combined_df['volatility_cycle_pos'] = bb_width_pr.apply(self.indicators_manager.derive_volatility_cycle_position)
            
            logger.debug(f"Calculated 'volatility_cycle_pos' (BB_WIDTH Percentile Rank mapped to [0, 2*PI]) for {symbol_market}.")
        else:
            logger.warning(f"'BB_WIDTH' column not found in complete_features for {symbol_market}. Cannot calculate 'volatility_cycle_pos'. Assigning default PI.")
            if not combined_df.empty: # Ensure column exists even if BB_WIDTH was missing
                 combined_df['volatility_cycle_pos'] = self.indicators_manager.derive_volatility_cycle_position(None) # Results in PI
            elif ohlcv_to_process.empty: # If combined_df is empty because ohlcv_to_process was empty
                pass # No data to add the column to
            else: # combined_df is empty but ohlcv_to_process was not (edge case, means existing_complete_df was also empty)
                logger.debug(f"Combined_df for {symbol_market} is initially empty, 'volatility_cycle_pos' will be added if data appears.")
        # === END BB_WIDTH Percentile Rank Calculation ===

        # === START R_DVU Calculation ===
        if not combined_df.empty:
            required_rdvu_cols = [
                'atr_current_rdvu_input', 
                'atr_average_rdvu_input', 
                'volatility_cycle_pos', 
                'base_stability_index_rdvu_input'
            ]
            if all(col in combined_df.columns for col in required_rdvu_cols):
                try:
                    def calculate_rdvu_for_row(row):
                        atr_curr = row['atr_current_rdvu_input']
                        atr_avg = row['atr_average_rdvu_input']
                        vol_cycle_pos = row['volatility_cycle_pos']
                        base_stab_idx = row['base_stability_index_rdvu_input']
                        
                        if pd.isna(atr_curr) or pd.isna(atr_avg) or pd.isna(vol_cycle_pos) or pd.isna(base_stab_idx):
                            return np.nan 
                        
                        return self.indicators_manager.calculate_rdvu(
                            atr_current=atr_curr,
                            atr_average=atr_avg,
                            volatility_cycle_position=vol_cycle_pos,
                            base_stability_index=base_stab_idx
                        )

                    combined_df['R_DVU'] = combined_df.apply(calculate_rdvu_for_row, axis=1)
                    combined_df['R_DVU'].fillna(0.0, inplace=True) 
                    logger.debug(f"Calculated 'R_DVU' for {symbol_market}.")

                except Exception as e:
                    logger.error(f"Error calculating R_DVU for {symbol_market}: {e}", exc_info=True)
                    if 'R_DVU' not in combined_df.columns: 
                        combined_df['R_DVU'] = 0.0 
                    else: 
                        combined_df['R_DVU'].fillna(0.0, inplace=True)
            else:
                missing_cols = [col for col in required_rdvu_cols if col not in combined_df.columns]
                logger.warning(f"Missing required columns for R_DVU calculation in {symbol_market}: {missing_cols}. Assigning default 0.0 to R_DVU.")
                combined_df['R_DVU'] = 0.0
        elif 'R_DVU' not in combined_df.columns and not ohlcv_to_process.empty : 
            if not combined_df.empty: # Ensure column exists even if BB_WIDTH was missing
                 combined_df['R_DVU'] = 0.0
                 logger.debug(f"Initialized 'R_DVU' to 0.0 for {symbol_market} as it was processed from an empty start.")
        # === END R_DVU Calculation ===

        self.complete_features_history[symbol_market] = combined_df.iloc[-self.max_history_length:]
        
        logger.debug(f"Complete features for {symbol_market} updated. Total: {len(self.complete_features_history[symbol_market])} bars.")

        # === START REGIME MERGE ===
        current_complete_df_for_merge = self.complete_features_history[symbol_market]
        if symbol_market in self.market_regime_history and not self.market_regime_history[symbol_market].empty:
            regime_series = self.market_regime_history[symbol_market].rename('market_regime')
            
            # Harmonize timezones before joining
            if current_complete_df_for_merge.index.tz is not None and regime_series.index.tz is None:
                logger.debug(f"Localizing regime_series index to {current_complete_df_for_merge.index.tz} for {symbol_market}")
                regime_series.index = regime_series.index.tz_localize(current_complete_df_for_merge.index.tz, ambiguous='infer', nonexistent='shift_forward')
            elif current_complete_df_for_merge.index.tz is None and regime_series.index.tz is not None:
                logger.debug(f"Converting regime_series index to naive for {symbol_market}")
                regime_series.index = regime_series.index.tz_convert(None)
            elif current_complete_df_for_merge.index.tz is not None and regime_series.index.tz is not None and current_complete_df_for_merge.index.tz != regime_series.index.tz:
                logger.debug(f"Converting regime_series index from {regime_series.index.tz} to {current_complete_df_for_merge.index.tz} for {symbol_market}")
                regime_series.index = regime_series.index.tz_convert(current_complete_df_for_merge.index.tz)
 
            current_complete_df_for_merge = current_complete_df_for_merge.join(regime_series, how='left')
            current_complete_df_for_merge['market_regime'] = current_complete_df_for_merge['market_regime'].ffill().bfill().fillna("Indeterminate")
            logger.debug(f"Market regime merged for {symbol_market}. First few regimes: {current_complete_df_for_merge['market_regime'].head().tolist()}")
        elif not current_complete_df_for_merge.empty:
            current_complete_df_for_merge['market_regime'] = "Indeterminate"
            logger.warning(f"Market regime history is empty or not found for {symbol_market}. 'market_regime' column initialized to 'Indeterminate'.")
        self.complete_features_history[symbol_market] = current_complete_df_for_merge
        # === END REGIME MERGE ===
 
        df_to_merge_sentiment = self.complete_features_history[symbol_market]
        if symbol_market in self.sentiment_history and self.sentiment_history[symbol_market]:
            relevant_sentiment_timestamps = {
                ts: score for ts, score in self.sentiment_history[symbol_market].items()
                if pd.notna(ts) and (current_processing_timestamp is None or ts <= current_processing_timestamp)
            }
            if relevant_sentiment_timestamps:
                sentiment_series = pd.Series(relevant_sentiment_timestamps, name='sentiment_score')
                sentiment_series.index.name = 'timestamp'
                
                # Harmonize timezones for sentiment series
                if df_to_merge_sentiment.index.tz is not None and sentiment_series.index.tz is None:
                    logger.debug(f"Localizing sentiment_series index to {df_to_merge_sentiment.index.tz} for {symbol_market}")
                    sentiment_series.index = sentiment_series.index.tz_localize(df_to_merge_sentiment.index.tz, ambiguous='infer', nonexistent='shift_forward')
                elif df_to_merge_sentiment.index.tz is None and sentiment_series.index.tz is not None:
                    logger.debug(f"Converting sentiment_series index to naive for {symbol_market}")
                    sentiment_series.index = sentiment_series.index.tz_convert(None)
                elif df_to_merge_sentiment.index.tz is not None and sentiment_series.index.tz is not None and df_to_merge_sentiment.index.tz != sentiment_series.index.tz:
                    logger.debug(f"Converting sentiment_series index from {sentiment_series.index.tz} to {df_to_merge_sentiment.index.tz} for {symbol_market}")
                    sentiment_series.index = sentiment_series.index.tz_convert(df_to_merge_sentiment.index.tz)
 
                df_to_merge_sentiment = df_to_merge_sentiment.join(sentiment_series, how='left')
                logger.debug(f"Sentiment scores merged for {symbol_market}. First few sentiment scores: {df_to_merge_sentiment['sentiment_score'].head().tolist()}")
            else:
                logger.debug(f"Pas de scores de sentiment pertinents à joindre pour {symbol_market} à {current_processing_timestamp if current_processing_timestamp else 'tous les temps'}")
        
        if 'sentiment_score' not in df_to_merge_sentiment.columns and not df_to_merge_sentiment.empty:
            df_to_merge_sentiment['sentiment_score'] = np.nan
            logger.warning(f"Sentiment score column not found for {symbol_market}. Initialized to NaN.")
        
        self.complete_features_history[symbol_market] = df_to_merge_sentiment
        logger.debug(f"Final complete features for {symbol_market} after sentiment merge. Total: {len(self.complete_features_history[symbol_market])} bars.")


    def get_fq_sequence(self, symbol_market: str, current_timestamp: pd.Timestamp, 
                        feature_scaler: Any, feature_columns: List[str]) -> Optional[np.ndarray]:
        self._init_asset_storage(symbol_market)
        self.update_complete_features(symbol_market, current_timestamp) 

        df_complete = self.complete_features_history.get(symbol_market)
        if df_complete is None or df_complete.empty:
            logger.warning(f"Complete features history is empty for {symbol_market}. Cannot generate FQ sequence.")
            return None
            
        df_relevant_history = df_complete[df_complete.index <= current_timestamp]

        if len(df_relevant_history) < self.fq_window_size:
            logger.warning(f"Not enough relevant data ({len(df_relevant_history)} < {self.fq_window_size}) "
                           f"for FQ sequence for {symbol_market} at {current_timestamp}")
            return None
        
        try:
            df_sequence = df_relevant_history.iloc[-self.fq_window_size:]
            
            final_sequence_data = pd.DataFrame(index=df_sequence.index)
            for col in feature_columns:
                if col in df_sequence.columns:
                    final_sequence_data[col] = df_sequence[col]
                else:
                    logger.warning(f"Feature column '{col}' missing for FQ sequence of {symbol_market}. Filling with 0.")
                    final_sequence_data[col] = 0.0
            
            if final_sequence_data.isnull().values.any():
                logger.warning(f"NaNs found in FQ sequence for {symbol_market}. Filling with 0.")
                final_sequence_data.fillna(0, inplace=True)
            
            values_to_scale = final_sequence_data[feature_columns].values

            if feature_scaler:
                sequence_scaled = feature_scaler.transform(values_to_scale)
            else:
                sequence_scaled = values_to_scale
                logger.warning(f"No scaler for FQ sequence of {symbol_market}. Using unscaled data.")
            
            return np.expand_dims(sequence_scaled, axis=0)
        except Exception as e:
            logger.error(f"Error building FQ sequence for {symbol_market}: {e}", exc_info=True)
            return None

    def get_latest_features(self, symbol_market: str) -> Optional[pd.Series]:
        self._init_asset_storage(symbol_market)
        latest_ohlcv_ts = self.historical_ohlcv.get(symbol_market, pd.DataFrame()).index.max()
        if pd.notna(latest_ohlcv_ts):
            self.update_complete_features(symbol_market, latest_ohlcv_ts)
        
        df_complete = self.complete_features_history.get(symbol_market)
        if df_complete is None or df_complete.empty:
            logger.warning(f"No complete features available for {symbol_market} to get latest.")
            return None
        return df_complete.iloc[-1].copy()

    def get_ohlcv_history(self, symbol_market: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        self._init_asset_storage(symbol_market)
        df_history = self.historical_ohlcv.get(symbol_market)
        if df_history is None or df_history.empty: return None
        return df_history.iloc[-limit:].copy() if limit and limit > 0 else df_history.copy()

class DRMManager: 
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        logger.info("DRMManager initialized.")

    def calculate_trade_size(self, decision: Dict, account_balance: float, latest_price: float) -> float:
        if latest_price <= 0:
            logger.error("Cannot calculate trade size: latest_price is zero or negative.")
            return 0.0
        if account_balance <=0:
            logger.warning("Account balance is zero or negative. Cannot calculate trade size.")
            return 0.0

        risk_multiplier = decision.get('risk_multiplier', 1.0) 
        base_pos_size_pct = self.config.getfloat('trading', 'position_size_percent', fallback=DEFAULT_POSITION_SIZE_PERCENT)
        
        adjusted_pos_size_pct = base_pos_size_pct * risk_multiplier
        max_risk_pct = self.config.getfloat('trading', 'max_risk_per_trade_percent', fallback=0.05)
        adjusted_pos_size_pct = min(adjusted_pos_size_pct, max_risk_pct)
        
        min_effective_risk_pct = self.config.getfloat('trading', 'min_effective_risk_percent', fallback=0.001)
        if adjusted_pos_size_pct < min_effective_risk_pct:
            logger.info(f"Adjusted position size {adjusted_pos_size_pct:.4f} is below minimum effective risk {min_effective_risk_pct:.4f}. No trade.")
            return 0.0

        position_size_currency = account_balance * adjusted_pos_size_pct
        quantity = position_size_currency / latest_price
        
        logger.info(f"Calculated trade size for {decision['symbol_market']}: {quantity:.8f} units "
                    f"(RiskMult: {risk_multiplier:.2f}, EffSize%: {adjusted_pos_size_pct:.2%})")
        return quantity

class ExitStrategyManager:
    def __init__(self, config_manager: ConfigManager, 
                 order_manager_instance: core_order_manager.OrderManager, 
                 position_manager_instance: core_position_manager.PositionManager):
        self.config = config_manager
        self.order_manager = order_manager_instance
        self.position_manager = position_manager_instance
        logger.info("ExitStrategyManager initialized.")

    def manage_open_position_exits(self, api_call_func: Callable, symbol_market: str, latest_price: float):
        position = self.position_manager.get_position_state(symbol_market)
        if not position or position.get('qty', 0) <= 0 or position.get('status') != 'open':
            return

        entry_price = position['entry_price']
        current_sl = position['sl']
        current_tp = position.get('tp')
        position_side = position['side']
        position_qty = position['qty']
        current_sl_order_id = position.get('sl_order_id')

        self.position_manager.update_position_highest_price(symbol_market, latest_price)
        position = self.position_manager.get_position_state(symbol_market) 

        exit_strategy_params = position.get('exit_strategy', {}) 
        move_to_be_enabled = exit_strategy_params.get('move_to_be_enabled', 
                                                     self.config.getboolean('exit_strategy', 'move_to_be_enabled_default', fallback=True))

        if move_to_be_enabled and position.get('exit_strategy_flags', {}).get('be_triggered', False) is False:
            move_to_be_ratio = self.config.getfloat('exit_strategy', 'move_to_be_ratio', fallback=0.5)
            be_buffer_pct = self.config.getfloat('exit_strategy', 'be_buffer_percent', fallback=0.001)

            if position_side == 'long' and current_tp and current_tp > entry_price:
                tp_distance = current_tp - entry_price
                if latest_price >= entry_price + (tp_distance * move_to_be_ratio):
                    new_sl = entry_price * (1 + be_buffer_pct) 
                    if new_sl > (current_sl or 0):
                        if self.order_manager.update_stop_loss(api_call_func, symbol_market, new_sl, position_qty, current_sl_order_id): 
                            self.position_manager.update_position_sl(symbol_market, new_sl)
                            if 'exit_strategy_flags' not in position: position['exit_strategy_flags'] = {}
                            position['exit_strategy_flags']['be_triggered'] = True 
                            self.position_manager.update_position_field(symbol_market, 'exit_strategy_flags', position['exit_strategy_flags'])
                            logger.info(f"Move-to-BE triggered for LONG {symbol_market}. New SL: {new_sl:.4f}")
        
        position = self.position_manager.get_position_state(symbol_market) 
        current_sl = position['sl'] if position else current_sl
        
        trailing_stop_enabled = exit_strategy_params.get('trailing_stop_enabled',
                                                         self.config.getboolean('exit_strategy', 'trailing_stop_enabled_default', fallback=True))

        if trailing_stop_enabled:
            ts_activation_profit_pct = self.config.getfloat('exit_strategy', 'trailing_stop_activation_profit_pct', fallback=0.01)
            ts_distance_pct = self.config.getfloat('exit_strategy', 'trailing_stop_distance_pct', fallback=0.005)

            if position_side == 'long':
                highest_price_seen = position.get('highest_price_seen', entry_price)
                current_profit_pct = (highest_price_seen - entry_price) / entry_price if entry_price > 0 else 0
                
                if current_profit_pct >= ts_activation_profit_pct:
                    potential_new_sl = highest_price_seen * (1 - ts_distance_pct)
                    if potential_new_sl > (current_sl or 0): 
                         if self.order_manager.update_stop_loss(api_call_func, symbol_market, potential_new_sl, position_qty, current_sl_order_id): 
                            self.position_manager.update_position_sl(symbol_market, potential_new_sl)
                            logger.info(f"Trailing Stop updated for LONG {symbol_market}. New SL: {potential_new_sl:.4f}")


# --- Main TradingBot Class ---
class TradingBot:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._setup_logging()
        
        self.system_state = SystemStateManager()
        self.notification_service = NotificationService(self.config_manager)

        self.indicators_manager = indicators.IndicatorsManager(self.config_manager)
        self.feature_engineer = feature_engineer.FeatureEngineer(self.config_manager, self.indicators_manager)
        
        self.model_manager = ModelManager(self.config_manager, self.feature_engineer)
        
        self.historical_data_manager = HistoricalDataManager(
            self.config_manager, self.feature_engineer, self.indicators_manager,
            max_history_length=self.config_manager.getint('trading', 'max_history_bars', fallback=DEFAULT_MAX_HISTORY_BARS)
        )
        
        self.ms_analyzer = MarketStructureAnalyzer(self.config_manager)
        self.of_analyzer = OrderFlowAnalyzer(self.config_manager)
        self.sentiment_analyzer = SentimentAnalyzer(self.config_manager)
        self.strategy_handler = StrategyHybrid(self.config_manager, self.indicators_manager)
        
        self.exchange_client: Optional[ccxt.Exchange] = None 
        self.order_manager: Optional[core_order_manager.OrderManager] = None
        self.position_manager: Optional[core_position_manager.PositionManager] = None
        
        self.drm_manager = DRMManager(self.config_manager) 
        self.exit_strategy_manager: Optional[ExitStrategyManager] = None 
        
        self._initialize_exchange_and_trading_services()
        if self.order_manager and self.position_manager: 
             self.exit_strategy_manager = ExitStrategyManager(self.config_manager, self.order_manager, self.position_manager)


    def _setup_logging(self):
        log_level_str = self.config_manager.get('logging', 'level', fallback='INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s',
            handlers=[ logging.FileHandler(LOG_FILE_PATH, encoding='utf-8'), logging.StreamHandler() ]
        )
        logger.info(f"Logging initialized at level {log_level_str}.")

    def _initialize_exchange_and_trading_services(self):
        exchange_id = self.config_manager.get('exchange', 'exchange_id', fallback='binance')
        api_key = self.config_manager.get('exchange', 'api_key', fallback=os.getenv(f'{exchange_id.upper()}_API_KEY'))
        api_secret = self.config_manager.get('exchange', 'api_secret', fallback=os.getenv(f'{exchange_id.upper()}_API_SECRET'))
        password = self.config_manager.get('exchange', 'password', fallback=os.getenv(f'{exchange_id.upper()}_PASSWORD')) 
        
        if not api_key or not api_secret:
            logger.warning("API key or secret is missing. Exchange functionality will be limited/simulated.")

        try:
            exchange_class = getattr(ccxt, exchange_id)
            params = {
                'apiKey': api_key, 'secret': api_secret,
                'timeout': self.config_manager.getint('exchange', 'timeout_ms', fallback=30000),
                'enableRateLimit': True,
                'options': {} 
            }
            if password: params['password'] = password

            if self.config_manager.getboolean('exchange', 'use_sandbox', fallback=False):
                if hasattr(exchange_class, 'urls') and 'test' in exchange_class.urls:
                    params['urls'] = {'api': exchange_class.urls['test']}
                    logger.info("Using SANDBOX mode for the exchange.")
                else:
                    logger.warning(f"Sandbox mode configured but not available for {exchange_id}.")
            
            default_type = self.config_manager.get('exchange', 'default_type', fallback=None)
            if default_type: params['options']['defaultType'] = default_type


            self.exchange_client = exchange_class(params)
            self.exchange_client.load_markets() 
            logger.info(f"Exchange '{exchange_id}' initialized (DefaultType: {default_type or 'not set'}).")
            
            self.order_manager = core_order_manager.OrderManager(self.exchange_client, self.config_manager)
            self.position_manager = core_position_manager.PositionManager(self.exchange_client, self.config_manager)

        except AttributeError:
            logger.critical(f"Exchange ID '{exchange_id}' not found in ccxt.", exc_info=True)
            self.system_state.trigger_emergency_shutdown()
            self.notification_service.send_emergency_notification(
                0, f"Failed to find exchange {exchange_id}"
            )
        except Exception as e:
            logger.critical(f"Failed to initialize exchange '{exchange_id}': {e}", exc_info=True)
            self.system_state.trigger_emergency_shutdown()
            self.notification_service.send_emergency_notification(
                0, f"Failed to init exchange {exchange_id}: {e}"
            )
    
    def _api_call_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        if self.system_state.is_emergency_shutdown():
            logger.warning(f"API call {func.__name__} aborted due to emergency shutdown.")
            return None
 
        retries = 0
        delay = API_RETRY_DELAY_SECONDS
        last_exception = None
 
        while retries < MAX_API_RETRIES:
            try:
                result = func(*args, **kwargs)
                self.system_state.reset_consecutive_errors() 
                return result
            except ccxt.NetworkError as e: 
                last_exception = e
                logger.warning(f"API call {func.__name__} NetworkError (try {retries+1}/{MAX_API_RETRIES}): {e}. Retrying in {delay}s...")
            except ccxt.ExchangeError as e: 
                last_exception = e
                logger.error(f"API call {func.__name__} ExchangeError (try {retries+1}/{MAX_API_RETRIES}): {e}. Retrying in {delay}s...")
                if isinstance(e, (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.AuthenticationError, 
                                   ccxt.InvalidNonce, ccxt.PermissionDenied, ccxt.AccountSuspended)):
                    logger.error(f"Non-retryable CCXT error for {func.__name__}: {e}. Aborting retries.")
                    self.system_state.increment_consecutive_errors()
                    self._check_emergency_shutdown()
                    return None 
            except Exception as e: 
                last_exception = e
                logger.error(f"API call {func.__name__} generic error (try {retries+1}/{MAX_API_RETRIES}): {e}", exc_info=True)
            
            time.sleep(delay)
            retries += 1
            delay *= API_BACKOFF_FACTOR
        
        logger.error(f"API call {func.__name__} failed after {MAX_API_RETRIES} retries. Last error: {last_exception}")
        self.system_state.increment_consecutive_errors()
        self._check_emergency_shutdown()
        return None
 
 
    def _check_emergency_shutdown(self):
        if self.system_state.get_consecutive_errors() >= MAX_CONSECUTIVE_ERRORS_BEFORE_EMERGENCY:
            if not self.system_state.is_emergency_shutdown():
                self.system_state.trigger_emergency_shutdown()
                self.notification_service.send_emergency_notification(
                    self.system_state.get_consecutive_errors(), 
                    "Max consecutive errors reached."
                )
 
    def _log_heartbeat(self):
        current_time = time.time()
        if current_time - self.system_state.last_heartbeat_time >= HEARTBEAT_INTERVAL_SECONDS:
            active_pos_count = self.position_manager.get_active_positions_count() if self.position_manager else 0
            logger.info(f"HEARTBEAT: Bot operational. Active positions: {active_pos_count}. "
                        f"Errors: {self.system_state.get_consecutive_errors()}.")
            self.system_state.update_heartbeat_time()
 
    def _fetch_asset_ohlcv_data(self, symbol_market: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        symbol_for_ccxt = symbol_market.split('@')[0] if '@' in symbol_market else symbol_market
        
        logger.info(f"Fetching OHLCV for {symbol_for_ccxt} ({timeframe}), limit={limit}...")
        if not self.exchange_client: 
            logger.error("Exchange client not available for fetching OHLCV.")
            return None
        
        ohlcv_raw = self._api_call_with_retry(self.exchange_client.fetch_ohlcv, symbol_for_ccxt, timeframe, limit=limit)
        
        if not ohlcv_raw or len(ohlcv_raw) == 0 :
            logger.warning(f"No OHLCV data returned for {symbol_for_ccxt}.")
            return pd.DataFrame() 
        if len(ohlcv_raw) < 2 and limit >1: 
            logger.warning(f"Insufficient OHLCV data ({len(ohlcv_raw)} bars) for {symbol_for_ccxt} when {limit} requested.")
        
        df = pd.DataFrame(ohlcv_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True) 
        logger.info(f"Fetched {len(df)} OHLCV bars for {symbol_for_ccxt}.")
        return df
        
    def _process_asset_data_and_features(self, symbol_market: str, latest_ohlcv_bar_df: pd.DataFrame):
        if latest_ohlcv_bar_df.empty: return
 
        self.historical_data_manager.add_ohlcv_data(symbol_market, latest_ohlcv_bar_df)
        
        full_ohlcv_history = self.historical_data_manager.get_ohlcv_history(symbol_market)
        if full_ohlcv_history is None or full_ohlcv_history.empty: return
 
        latest_ts = latest_ohlcv_bar_df.index[-1]
        
        if self.indicators_manager.is_initialized() and len(full_ohlcv_history) >= 2:
            prev_close = full_ohlcv_history['close'].iloc[-2]
            latest_close = full_ohlcv_history['close'].iloc[-1]
            self.indicators_manager.update_ratings(symbol_market, latest_close, prev_close, latest_ts)
        
        ms_features = self.ms_analyzer.calculate_market_structure_features(full_ohlcv_history, self.config_manager.config, self.indicators_manager)
        self.historical_data_manager.add_ms_features(symbol_market, latest_ts, ms_features)
        
        of_features = self.of_analyzer.analyze_order_flow_for_period(None) 
        self.historical_data_manager.add_of_features(symbol_market, latest_ts, of_features)
 
        self.historical_data_manager.update_complete_features(symbol_market, latest_ts)
 
 
    def _get_predictions(self, symbol_market: str, current_timestamp: pd.Timestamp) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        xgb_probs, fq_ratios = None, None
        latest_features = self.historical_data_manager.get_latest_features(symbol_market)
        if latest_features is None: 
            logger.warning(f"Cannot get latest features for {symbol_market} at {current_timestamp}")
            return None, None
 
        if self.model_manager.get_xgb_model():
            xgb_probs = self.model_manager.predict_xgboost(latest_features)
        
        if self.model_manager.get_fq_model():
            all_feature_names = self.model_manager.get_all_feature_names()
            scaler = self.model_manager.get_feature_scaler()
            if all_feature_names:
                fq_sequence = self.historical_data_manager.get_fq_sequence(
                    symbol_market, current_timestamp, scaler, all_feature_names
                )
                if fq_sequence is not None:
                    fq_ratios = self.model_manager.predict_futurequant(fq_sequence)
        return xgb_probs, fq_ratios
 
    def _execute_trade_action(self, decision: Dict):
        action = decision.get('action', 'HOLD')
        symbol_market = decision['symbol_market']
        latest_price = decision['latest_price']
 
        if not all([self.order_manager, self.position_manager, self.drm_manager, self.exit_strategy_manager]):
            logger.error("Trading service managers not initialized. Cannot execute trade action.")
            return
 
        if action == 'HOLD':
            logger.info(f"Decision for {symbol_market}: HOLD.")
            self.exit_strategy_manager.manage_open_position_exits(self._api_call_with_retry, symbol_market, latest_price)
            return
 
        with self.system_state.trading_lock: 
            if action == 'BUY':
                if self.position_manager.get_active_positions_count() >= self.config_manager.getint('trading', 'max_concurrent_positions', fallback=DEFAULT_MAX_CONCURRENT_POSITIONS):
                    logger.warning(f"Max concurrent positions reached. BUY for {symbol_market} skipped.")
                    return
                account_balance = self.order_manager.get_account_balance() 
                quantity = self.drm_manager.calculate_trade_size(decision, account_balance, latest_price)
                if quantity <= 0: logger.info(f"Calculated quantity is {quantity} for {symbol_market}. No BUY order."); return
 
                tp, sl = decision.get('tp_price'), decision.get('sl_price')
                logger.info(f"Attempting BUY {symbol_market}: Qty={quantity:.8f}, TP={tp}, SL={sl}")
                success, order_id = self.order_manager.place_buy_order_with_tp_sl(self._api_call_with_retry, symbol_market, quantity, None, tp, sl)
                if success and order_id:
                    self.position_manager.add_position(symbol_market, latest_price, quantity, 'long', tp, sl, 
                                                       decision['setup_quality'], decision['risk_multiplier'], order_id,
                                                       exit_strategy_params=decision.get('exit_strategy', {}))
                else: logger.error(f"Failed to place BUY order or TP/SL for {symbol_market}.")
 
            elif action == 'SELL_TO_CLOSE_LONG':
                position_data = self.position_manager.get_position_state(symbol_market)
                pos_qty = position_data.get('qty', 0) if position_data else 0
                if pos_qty <= 0: logger.warning(f"No LONG position for {symbol_market} to close."); return
                
                logger.info(f"Attempting CLOSE LONG for {symbol_market}: Qty={pos_qty:.8f}")
                self.position_manager.mark_position_closure_pending(symbol_market) 
                success, order_id = self.order_manager.place_sell_order(self._api_call_with_retry, symbol_market, pos_qty)
                if success:
                    self.position_manager.remove_position(symbol_market, latest_price) 
                else: 
                    logger.error(f"Failed to CLOSE LONG for {symbol_market}. Position remains 'closure_pending'. Manual check needed.")
 
    def _run_asset_processing_cycle(self, symbol_market: str):
        logger.info(f"--- Processing asset: {symbol_market} ---")
        if not all([self.historical_data_manager, self.indicators_manager, self.model_manager, 
                    self.strategy_handler, self.position_manager, self.order_manager, self.exit_strategy_manager]):
            logger.error(f"Core managers not available for processing {symbol_market}. Skipping.")
            return
 
        timeframe = self.config_manager.get('trading', 'timeframe', fallback='1h')
        df_recent_ohlcv = self._fetch_asset_ohlcv_data(symbol_market, timeframe, limit=self.config_manager.getint('strategy_hybrid', 'atr_period_sl_tp', fallback=14) + 2) # Fetch enough for ATR + a couple more
        
        if df_recent_ohlcv is None or df_recent_ohlcv.empty:
            logger.warning(f"No new OHLCV data for {symbol_market}. Skipping cycle for this asset.")
            return
 
        if len(df_recent_ohlcv) >= 2 :
             latest_completed_bar_df = df_recent_ohlcv.iloc[[-2]] 
        elif len(df_recent_ohlcv) == 1:
             latest_completed_bar_df = df_recent_ohlcv.iloc[[-1]] 
             logger.warning(f"Only 1 bar fetched for {symbol_market}, might be incomplete. Processing anyway.")
        else: 
            logger.warning(f"Unexpectedly few bars ({len(df_recent_ohlcv)}) for {symbol_market}. Cannot determine latest completed bar.")
            return
 
        current_processing_ts = latest_completed_bar_df.index[0]
        latest_price = latest_completed_bar_df['close'].iloc[0]
        logger.info(f"Processing for {symbol_market} based on bar at {current_processing_ts}, Close: {latest_price}")
 
        self._process_asset_data_and_features(symbol_market, latest_completed_bar_df)
        xgb_probs, fq_ratios = self._get_predictions(symbol_market, current_processing_ts)
 
        pi_ratings = self.indicators_manager.get_ratings(symbol_market)
        sentiment_data = self.sentiment_analyzer.get_sentiment_data(symbol_market) # Get full sentiment data dict
        sentiment_score = sentiment_data.get('score') if sentiment_data else None # Extract score
 
        position_data = self.position_manager.get_position_state(symbol_market)
        pos_qty = position_data.get('qty', 0) if position_data else 0
        balance = self.order_manager.get_account_balance() if self.order_manager else 0.0
        pnl = self.position_manager.get_daily_pnl()
        
        ms_latest = self.historical_data_manager.ms_features_history.get(symbol_market, {}).get(current_processing_ts, {})
        of_latest = self.historical_data_manager.of_features_history.get(symbol_market, {}).get(current_processing_ts, {})
 
        atr_value_for_decision = None
        try:
            atr_period_sl_tp = self.config_manager.getint('strategy_hybrid', 'atr_period_sl_tp', fallback=14)
            # Use the full df_recent_ohlcv for ATR calculation as it has more history
            if not df_recent_ohlcv.empty and all(col in df_recent_ohlcv.columns for col in ['high', 'low', 'close']):
                if len(df_recent_ohlcv) >= atr_period_sl_tp: 
                    atr_series = self.indicators_manager.calculate_atr(df_recent_ohlcv['high'],
                                                                       df_recent_ohlcv['low'],
                                                                       df_recent_ohlcv['close'],
                                                                       timeperiod=atr_period_sl_tp)
                    if not atr_series.empty:
                        # Try to get ATR corresponding to the current_processing_ts from the series
                        # If current_processing_ts is not in atr_series.index (e.g., it's the very last bar used for next period)
                        # then take the latest available ATR value.
                        if current_processing_ts in atr_series.index:
                            atr_value_for_decision = atr_series.get(current_processing_ts)
                        elif not atr_series.dropna().empty:
                             atr_value_for_decision = atr_series.dropna().iloc[-1]
                             logger.debug(f"[{symbol_market}] ATR for {current_processing_ts} not directly found, using latest available: {atr_value_for_decision:.4f}")
 
 
                        if pd.isna(atr_value_for_decision):
                            last_valid_atr = atr_series.dropna().iloc[-1] if not atr_series.dropna().empty else None
                            if last_valid_atr is not None:
                                logger.warning(f"[{symbol_market}] ATR for current bar {current_processing_ts} is NaN. Using last valid ATR: {last_valid_atr:.4f}")
                                atr_value_for_decision = last_valid_atr
                            else:
                                logger.warning(f"[{symbol_market}] ATR for current bar {current_processing_ts} is NaN and no prior valid ATR found.")
                                atr_value_for_decision = None 
                else:
                    logger.warning(f"[{symbol_market}] Not enough data in df_recent_ohlcv (len: {len(df_recent_ohlcv)}) to calculate ATR with period {atr_period_sl_tp}. Need at least {atr_period_sl_tp} bars.")
            else:
                logger.warning(f"[{symbol_market}] df_recent_ohlcv is empty or missing HLC columns. Cannot calculate ATR.")
        except Exception as e:
            logger.error(f"[{symbol_market}] Error calculating ATR: {e}")
            atr_value_for_decision = None
 
        # Fetch complete features for the current processing timestamp to get market regime
        complete_features_latest = self.historical_data_manager.complete_features_history.get(symbol_market)
        market_regime = "Indeterminate" # Default
        if complete_features_latest is not None and not complete_features_latest.empty and current_processing_ts in complete_features_latest.index:
            if 'market_regime' in complete_features_latest.columns:
                market_regime = complete_features_latest.loc[current_processing_ts, 'market_regime']
            else:
                logger.warning(f"Market regime column not found for {symbol_market} at {current_processing_ts}")
        else:
            logger.warning(f"Could not fetch complete_features or {current_processing_ts} for market_regime for {symbol_market}")
 
 
        decision = self.strategy_handler.generate_trade_decision(
            symbol_market=symbol_market,
            current_ratings=pi_ratings,
            latest_price=latest_price,
            sentiment_score=sentiment_score, # Pass only the score
            xgb_probabilities=xgb_probs,
            fq_predicted_ratios=fq_ratios,
            market_structure_features=ms_latest,
            order_flow_features=of_latest,
            atr_value=atr_value_for_decision,
            current_position_qty=pos_qty,
            account_balance=balance,
            daily_pnl=pnl,
            market_regime=market_regime, # Pass market regime
            sentiment_data=sentiment_data # Pass full sentiment data for context
        )
        self._execute_trade_action(decision)
        logger.info(f"--- Finished processing asset: {symbol_market} ---")
 
    def _process_asset_with_timeout(self, symbol_market: str, timeout: float):
        thread = threading.Thread(target=self._run_asset_processing_cycle, args=(symbol_market,), daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.error(f"TIMEOUT: Processing for {symbol_market} exceeded {timeout}s. Asset cycle skipped.")
            self.system_state.increment_consecutive_errors()
            self._check_emergency_shutdown()
 
    def trading_loop(self):
        if not self._check_system_health(initial_check=True): 
            logger.critical("Initial system health check failed. Aborting.")
            self.system_state.trigger_emergency_shutdown()
            self.notification_service.send_emergency_notification(0, "Initial system health check failed.")
            return
 
        assets_str = self.config_manager.get('trading', 'assets', fallback='')
        assets = [s.strip() for s in assets_str.split(',') if s.strip()]
        if not assets: logger.error("No assets configured. Stopping."); return
        
        update_interval = self.config_manager.getint('trading', 'update_interval_seconds', fallback=DEFAULT_UPDATE_INTERVAL_SECONDS)
        logger.info(f"Starting trading loop for {assets}. Update interval: {update_interval}s.")
 
        while not self.system_state.is_emergency_shutdown():
            cycle_start_time = time.time()
            self._log_heartbeat()
 
            if not self._check_exchange_connectivity():
                logger.error("Exchange connectivity lost during loop. Attempting reinitialization...")
                if not self._reinitialize_exchange_and_trading_services():
                    logger.critical("Failed to reinitialize exchange. Triggering emergency shutdown.")
                    self.system_state.trigger_emergency_shutdown()
                    self.notification_service.send_emergency_notification(
                        self.system_state.get_consecutive_errors(), 
                        "Exchange reinitialization failed."
                    )
                    break 
                else: logger.info("Exchange reinitialized successfully.")
            
            if self.position_manager and self.exit_strategy_manager:
                open_positions_symbols = list(self.position_manager.get_all_open_positions().keys()) 
                for sym_market_open in open_positions_symbols:
                    if self.system_state.is_emergency_shutdown(): break
                    df_price_check = self._fetch_asset_ohlcv_data(sym_market_open, 
                                                                  self.config_manager.get('trading', 'timeframe', fallback='1h'), 
                                                                  limit=1) 
                    if df_price_check is not None and not df_price_check.empty:
                        latest_price_for_exit = df_price_check['close'].iloc[-1]
                        self.exit_strategy_manager.manage_open_position_exits(self._api_call_with_retry, sym_market_open, latest_price_for_exit)
                    else:
                        logger.warning(f"Could not fetch latest price for {sym_market_open} for exit management.")
 
            if self.system_state.is_emergency_shutdown(): break
 
            for asset in assets:
                if self.system_state.is_emergency_shutdown(): break
                try:
                    self._process_asset_with_timeout(asset, MAX_ASSET_PROCESSING_TIMEOUT_SECONDS)
                except Exception as e: 
                    logger.error(f"Unhandled error in asset processing launcher for {asset}: {e}", exc_info=True)
                    self.system_state.increment_consecutive_errors(); self._check_emergency_shutdown()
            
            if self.system_state.is_emergency_shutdown(): break
 
            if self.position_manager: self.position_manager.save_positions_state()
            if self.indicators_manager: self.indicators_manager.save_ratings_state()
            if self.sentiment_analyzer: self.sentiment_analyzer.save_sentiment_state()
 
 
            elapsed_time = time.time() - cycle_start_time
            sleep_duration = max(0, update_interval - elapsed_time)
            logger.info(f"Cycle completed in {elapsed_time:.2f}s. Sleeping for {sleep_duration:.2f}s.")
            self._wait_with_shutdown_check(sleep_duration)
        
        logger.critical("Trading loop terminated.")
        if self.position_manager: self.position_manager.save_positions_state()
        if self.indicators_manager: self.indicators_manager.save_ratings_state()
        if self.sentiment_analyzer: self.sentiment_analyzer.save_sentiment_state()
 
 
    def _wait_with_shutdown_check(self, total_seconds: float, check_interval: float = 1.0): 
        start_wait = time.time()
        while time.time() - start_wait < total_seconds:
            if self.system_state.is_emergency_shutdown():
                logger.info("Wait interrupted by emergency shutdown.")
                break
            time_to_sleep = min(check_interval, total_seconds - (time.time() - start_wait))
            if time_to_sleep <=0: break 
            time.sleep(time_to_sleep)
 
 
    def _check_exchange_connectivity(self) -> bool:
        if not self.exchange_client: 
            logger.warning("Exchange client not initialized, cannot check connectivity.")
            return False
        status = self._api_call_with_retry(self.exchange_client.fetch_status)
        is_connected = status is not None and status.get('status') == 'ok'
        if not is_connected:
            logger.warning(f"Exchange connectivity check failed. Status: {status}")
        return is_connected
            
    def _reinitialize_exchange_and_trading_services(self) -> bool:
        logger.warning("Attempting to re-initialize exchange and dependent trading services...")
        self.exchange_client = None
        self.order_manager = None
        self.position_manager = None
        self.exit_strategy_manager = None 
 
        try:
            self._initialize_exchange_and_trading_services() 
            
            if self.exchange_client and self._check_exchange_connectivity():
                logger.info("Exchange and core trading services re-initialized successfully.")
                if self.order_manager and self.position_manager:
                    self.exit_strategy_manager = ExitStrategyManager(self.config_manager, self.order_manager, self.position_manager)
                
                if self.position_manager: 
                    self.position_manager.fetch_current_positions_from_exchange(self._api_call_with_retry)
                return True
            else:
                logger.error("Failed to establish connectivity after re-initializing exchange client.")
                return False
        except Exception as e:
            logger.critical(f"Critical error during re-initialization: {e}", exc_info=True)
            return False
 
    def _check_system_health(self, initial_check: bool = False) -> bool:
        logger.info("Performing system health check...")
        ok = True
        if not self.config_manager: ok=False; logger.error("FAIL: ConfigManager not loaded.")
        
        if not self.exchange_client: ok=False; logger.error("FAIL: ExchangeClient not initialized.")
        elif not self._check_exchange_connectivity(): ok=False; logger.error("FAIL: Exchange not connected.")
        
        critical_managers = [
            ("IndicatorMgr", self.indicators_manager), 
            ("FeatureEngMgr", self.feature_engineer),
            ("HistoricalDataMgr", self.historical_data_manager),
            ("ModelMgr", self.model_manager), 
            ("OrderMgr", self.order_manager), 
            ("PositionMgr", self.position_manager),
            ("StrategyHandler", self.strategy_handler),
            ("DRMMgr", self.drm_manager),
            ("ExitStrategyMgr", self.exit_strategy_manager),
            ("SentimentAnalyzer", self.sentiment_analyzer)
        ]
        for name, mgr_instance in critical_managers:
            if not mgr_instance: 
                ok=False; logger.error(f"FAIL: {name} not initialized.")
            elif hasattr(mgr_instance, 'is_initialized') and callable(getattr(mgr_instance, 'is_initialized')):
                if not mgr_instance.is_initialized():
                     ok=False; logger.error(f"FAIL: {name} not properly initialized (is_initialized() is False).")
        
        if initial_check:
            try: 
                log_dir = os.path.dirname(LOG_FILE_PATH)
                if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir) 
                with open(LOG_FILE_PATH, 'a') as f: f.write(f"{pd.Timestamp.now(tz='UTC')} - Health check log write test.\n")
            except Exception as e: 
                ok=False; logger.error(f"FAIL: Log file/dir not writable: {e}")
        
        logger.info(f"System Health Check {'PASS' if ok else 'FAIL'}")
        return ok
 
    def initial_data_load(self):
        logger.info("Starting initial historical data load...")
        assets_str = self.config_manager.get('trading', 'assets', fallback='')
        assets = [s.strip() for s in assets_str.split(',') if s.strip()]
        timeframe = self.config_manager.get('trading', 'timeframe', fallback='1h')
        hist_limit = self.config_manager.getint('trading', 'max_history_bars', fallback=DEFAULT_MAX_HISTORY_BARS)
 
        for asset in assets:
            if self.system_state.is_emergency_shutdown(): break
            logger.info(f"Loading initial data for {asset}...")
            # Fetch more data initially to allow for proper calculation of rolling indicators like ATR, BBW percentile
            # For example, if bbw_percentile_window is 100, and ATR period is 14, need at least 100 + some buffer.
            # Let's use max_history_bars + bbw_percentile_window as a safe bet, capped by a reasonable max like 1000
            bbw_percentile_window = self.config_manager.getint('trinary_config', 'trinary_bbw_percentile_window', fallback=100)
            atr_period_sl_tp = self.config_manager.getint('strategy_hybrid', 'atr_period_sl_tp', fallback=14)
            required_bars_for_calcs = max(bbw_percentile_window, atr_period_sl_tp) 
            effective_hist_limit = max(hist_limit, required_bars_for_calcs + 50) # Add buffer
            
            df_ohlcv = self._fetch_asset_ohlcv_data(asset, timeframe, limit=effective_hist_limit)
            if df_ohlcv is not None and not df_ohlcv.empty:
                self.historical_data_manager.add_ohlcv_data(asset, df_ohlcv)
                full_hist_df = self.historical_data_manager.get_ohlcv_history(asset)
                if full_hist_df is not None and not full_hist_df.empty:
                    if self.indicators_manager.is_initialized():
                        for i in range(1, len(full_hist_df)):
                            self.indicators_manager.update_ratings(
                                asset, 
                                full_hist_df['close'].iloc[i], 
                                full_hist_df['close'].iloc[i-1], 
                                full_hist_df.index[i]
                            )
                    for ts_idx in full_hist_df.index:
                        # For historical build-up, MS features should consider history *up to* ts_idx
                        history_slice_for_ms = full_hist_df.loc[:ts_idx]
                        if len(history_slice_for_ms) < self.ms_analyzer.min_bars_for_analysis if hasattr(self.ms_analyzer, 'min_bars_for_analysis') else 20: # Check if enough data for MS
                            ms_f = {} # Not enough data, provide empty features
                        else:
                            ms_f = self.ms_analyzer.calculate_market_structure_features(history_slice_for_ms, self.config_manager.config, self.indicators_manager)
                        self.historical_data_manager.add_ms_features(asset, ts_idx, ms_f)
                        
                        # OF features are conceptual/placeholder in this version, so pass None or minimal data
                        of_f = self.of_analyzer.analyze_order_flow_for_period(None) 
                        self.historical_data_manager.add_of_features(asset, ts_idx, of_f)
 
                        # Sentiment for historical data might be complex. For now, assume no historical sentiment scores.
                        # If available, they would be added via add_sentiment_score per timestamp.
                        # Here, we are focusing on features derived from OHLCV.
 
                    # Build all complete features once underlying data (OHLCV, TA, MS, OF) is populated
                    # This call will iterate internally and build features, including R_DVU and volatility_cycle_pos
                    self.historical_data_manager.update_complete_features(asset) 
 
                hist_data = self.historical_data_manager.get_ohlcv_history(asset)
                logger.info(f"Initial data for {asset} processed. {len(hist_data) if hist_data is not None and not hist_data.empty else 0} bars.")
            else: logger.warning(f"Failed to load initial data for {asset}.")
        
        if self.sentiment_analyzer: self.sentiment_analyzer.load_sentiment_state() # Load any persisted sentiment
 
        logger.info("Initial data load complete.")
 
def main():
    bot = None
    try:
        if not os.path.exists('config') and os.path.basename(CONFIG_FILE_PATH) == CONFIG_FILE_PATH:
            logger.warning("Consider placing config.ini in a 'config/' subdirectory.")
 
        config_mgr = ConfigManager(CONFIG_FILE_PATH) 
        bot = TradingBot(config_mgr)
        
        if bot.system_state.is_emergency_shutdown():
             logger.critical("Bot initialization failed or led to emergency state. Exiting.")
             return
 
        bot.initial_data_load()
        if bot.system_state.is_emergency_shutdown():
             logger.critical("Emergency state after initial data load. Exiting.")
             return
             
        if bot.position_manager: 
            bot.position_manager.fetch_current_positions_from_exchange(bot._api_call_with_retry)
 
        bot.trading_loop()
 
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Configuration file problem. {e}") 
        logging.getLogger("MainFallback").critical(f"Configuration file problem. {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Trading bot interrupted by user (Ctrl+C). Shutting down gracefully...")
        if bot and bot.system_state: bot.system_state.trigger_emergency_shutdown() 
    except Exception as e:
        logger.critical(f"Unhandled critical error in main execution: {e}", exc_info=True)
        if bot and bot.notification_service and bot.system_state: 
            if not bot.system_state.is_emergency_shutdown(): 
                bot.system_state.trigger_emergency_shutdown() 
            bot.notification_service.send_emergency_notification(
                bot.system_state.get_consecutive_errors() if bot.system_state else 999, 
                f"Unhandled critical error in main: {e}\n{traceback.format_exc()}"
            )
    finally:
        logger.info("Trading bot application attempting to finalize...")
        if bot:
            if bot.position_manager: bot.position_manager.save_positions_state() # Already self.position_manager
            if bot.indicators_manager: bot.indicators_manager.save_ratings_state()
            if bot.sentiment_analyzer: bot.sentiment_analyzer.save_sentiment_state()
        logger.info("Trading bot application finished.")
        logging.shutdown()
 
if __name__ == "__main__":
    log_dir_main = os.path.dirname(LOG_FILE_PATH)
    if log_dir_main and not os.path.exists(log_dir_main):
        try:
            os.makedirs(log_dir_main)
        except OSError as e:
            print(f"Error creating log directory {log_dir_main}: {e}")
    main()