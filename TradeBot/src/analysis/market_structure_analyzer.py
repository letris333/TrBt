# market_structure_analyzer.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple, Any
from configparser import ConfigParser
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from .. import indicators # Import the module to access the class

logger = logging.getLogger(__name__)

class MarketStructureAnalyzer:
    def __init__(self, config_manager: ConfigParser):
        self.config = config_manager
        self.min_bars_for_analysis = self.config.getint('market_structure', 'min_bars_for_analysis', fallback=20)
        logger.info("MarketStructureAnalyzer initialized.")

    def calculate_volume_profile_slice(self, df_ohlcv_volume: pd.DataFrame, price_low: float, price_high: float, resolution: float) -> Dict[float, float]:
        """
        Calcule le profil de volume pour une tranche de prix donnée.
        Retourne un dict {price_bin_center: volume}.
        """
        if df_ohlcv_volume is None or df_ohlcv_volume.empty or resolution <= 0:
            return {}

        df_slice = df_ohlcv_volume[(df_ohlcv_volume['close'] >= price_low) & (df_ohlcv_volume['close'] <= price_high)].copy()

        if df_slice.empty:
            return {}

        min_price = price_low
        max_price = price_high
        bins = np.arange(min_price, max_price + resolution * 0.5, resolution)

        df_slice['price_bin'] = np.floor(df_slice['close'] / resolution) * resolution
        volume_profile = df_slice.groupby('price_bin')['volume'].sum()

        return volume_profile.to_dict()

    def calculate_market_structure_features(
        self,
        df_historical_ohlcv_volume: pd.DataFrame,
        config: ConfigParser,
        indicators_manager_instance: Any
    ) -> Dict:
        """
        Calcule les features de structure de marché (Volume Profile, POC, VA, HVN, LVN, etc.)
        pour la fin du DataFrame (temps 't').
        """
        features_ms = {}

        if df_historical_ohlcv_volume is None or df_historical_ohlcv_volume.empty:
            logger.warning("Données historiques vides pour le calcul MS.")
            return features_ms

        try:
            vp_timeframe_days = config.getint('market_structure', 'vp_timeframe_days', fallback=1)
            vp_resolution = config.getfloat('market_structure', 'vp_resolution', fallback=10)
            hvn_vol_threshold_percent = config.getfloat('market_structure', 'hvn_volume_threshold_percent', fallback=5)
            lvn_vol_threshold_percent = config.getfloat('market_structure', 'lvn_volume_threshold_percent', fallback=0.5)
            ms_level_lookback_bars = config.getint('market_structure', 'ms_level_lookback_bars', fallback=50)
            ms_volatility_atr_period = config.getint('market_structure', 'ms_volatility_atr_period', fallback=14)

        except Exception as e:
            logger.error(f"Erreur de configuration [market_structure]: {e}. Utilisation des fallbacks.")
            vp_timeframe_days = 1
            vp_resolution = 10
            hvn_vol_threshold_percent = 5
            lvn_vol_threshold_percent = 0.5
            ms_level_lookback_bars = 50
            ms_volatility_atr_period = 14

        end_time = df_historical_ohlcv_volume.index[-1]
        start_time_vp = end_time - timedelta(days=vp_timeframe_days)
        df_vp_window = df_historical_ohlcv_volume[df_historical_ohlcv_volume.index >= start_time_vp].copy()

        if df_vp_window.empty:
            logger.warning(f"Fenêtre de données VP vide pour {end_time} ({vp_timeframe_days} jours lookback).")
            if len(df_historical_ohlcv_volume) > 1:
                logger.warning("VP window is empty, calculating on available historical data.")
                df_vp_window = df_historical_ohlcv_volume.copy()
            else:
                return features_ms

        required_cols = ['close', 'volume', 'high', 'low']
        if not all(col in df_vp_window.columns for col in required_cols):
            logger.error(f"Colonnes {required_cols} manquantes pour le calcul MS.")
            return features_ms

        latest_close_price_in_window = df_vp_window['close'].iloc[-1]
        atr_series_vp_window = pd.Series(dtype=float)
        current_atr_value = np.nan
        current_atr_normalized = np.nan

        if len(df_vp_window) >= ms_volatility_atr_period:
            try:
                atr_series_vp_window = indicators_manager_instance.calculate_atr(
                    df_vp_window['high'],
                    df_vp_window['low'],
                    df_vp_window['close'],
                    timeperiod=ms_volatility_atr_period
                )
                if not atr_series_vp_window.empty and not pd.isna(atr_series_vp_window.iloc[-1]):
                    current_atr_value = atr_series_vp_window.iloc[-1]
                    if latest_close_price_in_window > 0:
                        current_atr_normalized = current_atr_value / latest_close_price_in_window
            except Exception as e:
                logger.warning(f"Error calculating ATR for MS context: {e}")
        else:
            logger.warning(f"Not enough data in df_vp_window (len: {len(df_vp_window)}) to calculate ATR with period {ms_volatility_atr_period}.")

        min_price_window = df_vp_window['low'].min()
        max_price_window = df_vp_window['high'].max()

        volume_profile_dict = self.calculate_volume_profile_slice(df_vp_window, min_price_window, max_price_window, vp_resolution)
        volume_profile_series = pd.Series(volume_profile_dict).sort_index()

        if volume_profile_series.empty:
            logger.warning("Volume Profile calculé est vide.")
            return features_ms

        total_volume_vp = volume_profile_series.sum()
        if total_volume_vp == 0:
            logger.warning("Volume total dans la fenêtre VP est zéro.")
            return features_ms

        poc_price_bin = volume_profile_series.idxmax()
        features_ms['ms_poc_price'] = float(poc_price_bin)

        volume_profile_percent = volume_profile_series / total_volume_vp

        sorted_volume_bins = volume_profile_percent.sort_values(ascending=False)
        cumulative_volume_percent = 0
        va_bins = []

        va_bins.append(poc_price_bin)
        cumulative_volume_percent += sorted_volume_bins.loc[poc_price_bin]

        va_bins_by_volume = sorted_volume_bins.index.tolist()
        va_prices = [poc_price_bin]
        current_volume_sum = volume_profile_series.loc[poc_price_bin]

        for bin_price in va_bins_by_volume:
            if bin_price == poc_price_bin:
                continue
            if current_volume_sum >= 0.70:
                break

            va_prices.append(bin_price)
            current_volume_sum += volume_profile_series.loc[bin_price]

        if va_prices:
            va_low_price = min(va_prices)
            va_high_price = max(va_prices)
            features_ms['ms_va_low_price'] = float(va_low_price)
            features_ms['ms_va_high_price'] = float(va_high_price)
        else:
            features_ms['ms_va_low_price'] = np.nan
            features_ms['ms_va_high_price'] = np.nan
            logger.warning("Volume Area could not be calculated.")

        hvn_prices = []
        lvn_prices = []
        hvn_vol_threshold = total_volume_vp * (hvn_vol_threshold_percent / 100)
        lvn_vol_threshold = total_volume_vp * (lvn_vol_threshold_percent / 100)

        price_bins_sorted = volume_profile_series.index.tolist()

        for i in range(len(price_bins_sorted)):
            bin_price = price_bins_sorted[i]
            bin_volume = volume_profile_series.loc[bin_price]

            is_local_peak = True
            if i > 0 and volume_profile_series.loc[price_bins_sorted[i-1]] >= bin_volume:
                is_local_peak = False
            if i < len(price_bins_sorted) - 1 and volume_profile_series.loc[price_bins_sorted[i+1]] >= bin_volume:
                is_local_peak = False

            if is_local_peak and bin_volume >= hvn_vol_threshold:
                hvn_prices.append(float(bin_price))

            is_local_valley = True
            if i > 0 and volume_profile_series.loc[price_bins_sorted[i-1]] <= bin_volume:
                is_local_valley = False
            if i < len(price_bins_sorted) - 1 and volume_profile_series.loc[price_bins_sorted[i+1]] <= bin_volume:
                is_local_valley = False

            if is_local_valley and bin_volume <= lvn_vol_threshold:
                lvn_prices.append(float(bin_price))

        features_ms['ms_hvn_prices'] = hvn_prices
        features_ms['ms_lvn_prices'] = lvn_prices

        features_ms['ms_vp_data_start_time'] = pd.Timestamp(start_time_vp).timestamp() if isinstance(start_time_vp, (datetime, pd.Timestamp)) else np.nan
        features_ms['ms_vp_data_end_time'] = pd.Timestamp(end_time).timestamp() if isinstance(end_time, (datetime, pd.Timestamp)) else np.nan
        features_ms['ms_vp_lookback_days'] = vp_timeframe_days
        features_ms['ms_formation_atr_value'] = float(current_atr_value) if not pd.isna(current_atr_value) else np.nan
        features_ms['ms_formation_atr_normalized'] = float(current_atr_normalized) if not pd.isna(current_atr_normalized) else np.nan
        features_ms['ms_formation_atr_period'] = ms_volatility_atr_period

        current_price = df_vp_window['close'].iloc[-1]

        features_ms['ms_price_vs_poc'] = (current_price - features_ms.get('ms_poc_price', current_price)) / current_price if features_ms.get('ms_poc_price') is not None and current_price > 0 else np.nan
        features_ms['ms_price_vs_va_low'] = (current_price - features_ms.get('ms_va_low_price', current_price)) / current_price if features_ms.get('ms_va_low_price') is not None and current_price > 0 else np.nan
        features_ms['ms_price_vs_va_high'] = (current_price - features_ms.get('ms_va_high_price', current_price)) / current_price if features_ms.get('ms_va_high_price') is not None and current_price > 0 else np.nan

        lookback_rejection_bars = 5
        recent_bars = df_vp_window.iloc[-lookback_rejection_bars-1:]

        if len(recent_bars) > 1 and features_ms.get('ms_va_high_price') is not None and features_ms.get('ms_va_low_price') is not None:
            va_high = features_ms['ms_va_high_price']
            va_low = features_ms['ms_va_low_price']

            features_ms['ms_recent_rejection_va_high'] = False
            if not recent_bars.iloc[:-1].empty:
                if ((recent_bars.iloc[:-1]['high'] > va_high) & (recent_bars.iloc[:-1]['close'] < va_high)).any():
                    features_ms['ms_recent_rejection_va_high'] = True

            features_ms['ms_recent_rejection_va_low'] = False
            if not recent_bars.iloc[:-1].empty:
                if ((recent_bars.iloc[:-1]['low'] < va_low) & (recent_bars.iloc[:-1]['close'] > va_low)).any():
                    features_ms['ms_recent_rejection_va_low'] = True

        else:
            features_ms['ms_recent_rejection_va_high'] = False
            features_ms['ms_recent_rejection_va_low'] = False
            if len(recent_bars) <= 1:
                logger.debug(f"Pas assez de barres récentes ({len(recent_bars)}) pour la vérification de rejet MS.")

        logger.debug(f"MS Features calculées pour {end_time}: {features_ms}")
        return features_ms

    def get_historical_of_data_for_period(
        self,
        df_all_of_data: Optional[pd.DataFrame],
        start_time: datetime,
        end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Extrait les données OF pour une période spécifique.
        CECI EST UN PLACEHOLDER - NÉCESSITE UNE SOURCE DE DONNÉES OF GLOBALE.
        """
        if df_all_of_data is None or df_all_of_data.empty:
            return None
        
        if df_all_of_data.index.tzinfo is None:
            logger.warning("L'index des données OF historiques n'est pas timezone aware. Supposons UTC.")
            df_all_of_data = df_all_of_data.tz_localize(timezone.utc)

        of_slice = df_all_of_data[(df_all_of_data.index >= start_time) & (df_all_of_data.index <= end_time)].copy()

        return of_slice if not of_slice.empty else None

    def initialize(self, config: ConfigParser):
        """
        Initialise le module d'analyse de structure de marché avec la configuration.
        
        Args:
            config: Objet ConfigParser contenant la configuration du système
        """
        logger.info("Initialisation du module de structure de marché")