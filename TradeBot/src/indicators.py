# indicators.py
"""
Module unifié pour les indicateurs techniques utilisés dans le système de trading hybride.
Contient à la fois les Pi-Ratings et les indicateurs techniques classiques (TA-Lib),
ainsi que les fonctions du système Trinary.
"""
import pandas as pd
import numpy as np
import talib
import logging
import json
import os
import math # Added for Pi and other math functions
from typing import Dict, Optional, Tuple, List, Any
from configparser import ConfigParser # Added import for ConfigParser

logger = logging.getLogger(__name__)

# Global store for indicator configurations, initialized by initialize_indicators
class IndicatorsManager:
    def __init__(self, config: ConfigParser):
        self.config = config
        self._indicator_config: Dict[str, Any] = {}
        self._pi_ratings_params: Dict[str, Any] = {}
        self._pi_ratings_state: Dict[str, Any] = {}
        self._pi_state_file: Optional[str] = None

        # Constants for Trinary System (can be overridden by config)
        self.DEFAULT_TRINARY_RSI_PERIOD = 14
        self.DEFAULT_TRINARY_ATR_PERIOD_STABILITY = 14
        self.DEFAULT_TRINARY_ATR_PERIOD_VOLATILITY = 14 # For RDVU's atr_current
        self.DEFAULT_TRINARY_ATR_SMA_PERIOD = 50      # For RDVU's atr_average
        self.DEFAULT_TRINARY_BB_PERIOD = 20
        self.DEFAULT_TRINARY_BB_STDDEV = 2.0
        self.DEFAULT_TRINARY_BBW_PERCENTILE_WINDOW = 100

        # Mathematical constant Pi
        self.PI = np.pi

        # Golden Ratio pour Pi-Ratings
        self.PHI = (1 + math.sqrt(5)) / 2
        self.LAMBDA = 1.0 / self.PHI
        self.GAMMA = 1.0 / (self.PHI * self.PHI)

        # Variables for Pi-Ratings state
        self._ratings_state: Dict[str, Dict[str, float]] = {}
        self._ratings_history: Dict[str, List[Dict]] = {}
        self._max_history_per_asset: int = 100
        self._decay_factor: float = 1.0
        self._max_rating_value: float = 10.0
        self._ratings_state_file: Optional[str] = None

        self._initialize_indicators_internal()

    def is_initialized(self) -> bool:
        """
        Checks if the IndicatorsManager has been initialized.
        """
        return bool(self._indicator_config) and bool(self._ratings_state_file)

    def _initialize_indicators_internal(self):
        """
        Initialise tous les indicateurs nécessaires au système:
        - Pi-Ratings: chargement de l'état, configuration des paramètres
        - TA-Lib: validation des dépendances
        - Charge la configuration des indicateurs pour les helpers.
        """
        self._indicator_config = self.config.get_section('indicators') if self.config.has_section('indicators') else {}
        logger.info(f"Indicator configuration loaded: {list(self._indicator_config.keys())}")

        self._initialize_ratings_state()

        try:
            talib_version = talib.get_version() if hasattr(talib, 'get_version') else "Inconnu"
            logger.info(f"TA-Lib initialisé (version: {talib_version})")
        except Exception as e:
            logger.warning(f"Problème avec l'initialisation de TA-Lib: {e}. Certains indicateurs TA ne fonctionneront pas.")


    def _initialize_ratings_state(self):
        """Initialise le chemin du fichier de state, les paramètres et charge le state si existant."""
        self._ratings_state_file = self.config.get('pi_ratings', 'ratings_state_file', fallback='data/pi_ratings_state.json')

        # Nouveaux paramètres de configuration
        self._decay_factor = self.config.getfloat('pi_ratings', 'decay_factor', fallback=0.999)
        self._max_history_per_asset = self.config.getint('pi_ratings', 'max_history_points', fallback=100)
        self._max_rating_value = self.config.getfloat('pi_ratings', 'max_rating_value', fallback=10.0)

        self.load_ratings_state()
        logger.info(f"Pi-Ratings state initialized. File: {self._ratings_state_file}, Decay: {self._decay_factor}, MaxHistory: {self._max_history_per_asset}, MaxRating: {self._max_rating_value}")

    def load_ratings_state(self):
        """Charge le dernier état des ratings depuis le fichier."""
        if self._ratings_state_file and os.path.exists(self._ratings_state_file):
            try:
                with open(self._ratings_state_file, 'r') as f:
                    data = json.load(f)
                    self._ratings_state = data.get('current_state', {})
                    self._ratings_history = data.get('history', {})

                    logger.info(f"Ratings state chargé depuis {self._ratings_state_file}.")
                    # Assurer que les valeurs sont float
                    for asset in self._ratings_state:
                        if 'R_H' in self._ratings_state[asset]:
                            self._ratings_state[asset]['R_H'] = float(self._ratings_state[asset]['R_H'])
                        if 'R_A' in self._ratings_state[asset]:
                             self._ratings_state[asset]['R_A'] = float(self._ratings_state[asset]['R_A'])
            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"Erreur lors du chargement des ratings state depuis {self._ratings_state_file}: {e}")
                self._ratings_state = {} # Revert to empty state on error
                self._ratings_history = {}
            except Exception as e:
                logger.error(f"Erreur inattendue lors du chargement des ratings state: {e}", exc_info=True)
                self._ratings_state = {}
                self._ratings_history = {}
        else:
            self._ratings_state = {}
            self._ratings_history = {}
            logger.info("Aucun fichier de ratings state trouvé. Initialisation à vide.")

    def save_ratings_state(self):
        """Sauvegarde l'état actuel des ratings dans le fichier."""
        if self._ratings_state_file:
            try:
                # Créer le répertoire si nécessaire
                dirname = os.path.dirname(self._ratings_state_file)
                if dirname and not os.path.exists(dirname):
                     os.makedirs(dirname)

                # Sauvegarder l'état courant et l'historique ensemble
                data_to_save = {
                    "current_state": self._ratings_state,
                    "history": self._ratings_history
                }

                with open(self._ratings_state_file, 'w') as f:
                    json.dump(data_to_save, f, indent=4)
                logger.debug(f"Ratings state sauvegardé dans {self._ratings_state_file}.")
            except (IOError, TypeError) as e:
                logger.error(f"Erreur lors de la sauvegarde des ratings state dans {self._ratings_state_file}: {e}")
            except Exception as e:
                logger.error(f"Erreur inattendue lors de la sauvegarde des ratings state: {e}", exc_info=True)


    # ===== PI-RATINGS FONCTIONS =====

    def get_ratings(self, symbol_market: str) -> Dict[str, float]:
        """Retourne les ratings R_H et R_A pour un actif."""
        # Initialiser si l'actif n'existe pas encore
        if symbol_market not in self._ratings_state:
            self._ratings_state[symbol_market] = {'R_H': 0.0, 'R_A': 0.0}
            self._ratings_history[symbol_market] = []
            logger.info(f"Initialisation des ratings pour {symbol_market} à 0.0.")
        return self._ratings_state[symbol_market]

    def get_ratings_history(self, symbol_market: str, limit: int = None) -> List[Dict]:
        """Retourne l'historique des ratings pour un actif.

        Args:
            symbol_market: Identifiant de l'actif
            limit: Nombre maximal de points à retourner, du plus récent au plus ancien

        Returns:
            Liste de dictionnaires contenant {'timestamp', 'R_H', 'R_A', 'price'}
        """
        if symbol_market not in self._ratings_history:
            return []

        history = self._ratings_history[symbol_market]
        if limit and len(history) > limit:
            return history[-limit:]
        return history

    def calculate_movement_magnitude(self, price_change_log: float) -> float: # Changed to accept log return
        """
        Calcule la magnitude pondérée du mouvement de prix.
        price_change_log est le log return (ln(P_new/P_old)).
        """
        # Utiliser la variation absolue pour la magnitude
        e = abs(price_change_log)

        # Appliquer la fonction de pondération logarithmique basée sur Phi
        if e > 0:
            m = self.PHI * np.log10(1 + e * self.PHI) # Use log10 as per original formula
        else:
            m = 0.0

        return m

    def apply_decay_to_ratings(self):
        """Applique une dégradation progressive à toutes les ratings pour les ramener vers 0."""
        if self._decay_factor >= 1.0:
            return  # Pas de dégradation

        for symbol_market in self._ratings_state:
            current = self._ratings_state[symbol_market]
            # Appliquer le facteur de dégradation à R_H et R_A
            if 'R_H' in current:
                current['R_H'] *= self._decay_factor
            if 'R_A' in current:
                current['R_A'] *= self._decay_factor


    def normalize_ratings(self, r_h: float, r_a: float, max_rating: float = None) -> Tuple[float, float]:
        """Normalise les ratings pour qu'elles restent dans une plage raisonnable.

        Args:
            r_h: Rating haussière
            r_a: Rating baissière
            max_rating: Valeur maximum autorisée (les ratings seront normalisées proportionnellement)

        Returns:
            Tuple (r_h_norm, r_a_norm) des ratings normalisées
        """
        if max_rating is None:
            max_rating = self._max_rating_value  # Utiliser la valeur globale

        # Ensure inputs are finite
        if not np.isfinite(r_h): r_h = 0.0
        if not np.isfinite(r_a): r_a = 0.0

        # If both ratings are within the max limit, return them as is
        # Use absolute values as ratings can potentially become negative in some variations,
        # although the update logic here prevents it by using max(0.0, ...)
        if abs(r_h) <= max_rating and abs(r_a) <= max_rating:
            return r_h, r_a

        # Calculate the maximum absolute value
        max_val = max(abs(r_h), abs(r_a))
        if max_val == 0:
            return 0.0, 0.0

        # Normalize proportionally
        scale_factor = max_rating / max_val
        return r_h * scale_factor, r_a * scale_factor


    def calculate_confidence_score(self, r_h: float, r_a: float) -> float:
        """
        Calcule un score de confiance unique basé sur les ratings R_H et R_A.

        Returns:
            Float entre -1 et 1 représentant le signal (positif = haussier, négatif = baissier)
        """
        # Ensure inputs are finite
        if not np.isfinite(r_h): r_h = 0.0
        if not np.isfinite(r_a): r_a = 0.0

        total = abs(r_h) + abs(r_a)
        # Avoid division by zero
        if total < 1e-9: # Use a small epsilon
            return 0.0

        return (r_h - r_a) / total


    def update_ratings(self, symbol_market: str, latest_close: float, previous_close: float, timestamp=None):
        """
        Met à jour les ratings R_H et R_A pour un actif basé sur la variation de prix.
        Applique la dégradation et ajoute à l'historique.

        Args:
            symbol_market: Identifiant de l'actif
            latest_close: Prix de clôture le plus récent
            previous_close: Prix de clôture précédent
            timestamp: Timestamp du latest_close (optionnel, utilise maintenant si None)
        """
        if previous_close is None or previous_close <= 0 or latest_close is None or latest_close <= 0:
            logger.warning(f"Prix invalides pour la mise à jour des ratings de {symbol_market}. Anciens: {previous_close}, Nouveaux: {latest_close}. Mise à jour sautée.")
            return

        # Obtenir le timestamp actuel si non fourni
        if timestamp is None:
            timestamp = pd.Timestamp.now().isoformat()
        elif isinstance(timestamp, pd.Timestamp):
             timestamp = timestamp.isoformat() # Ensure isoformat for JSON serialization


        # Calculer la variation de prix en pourcentage (décimal) ou log return
        try:
            price_change_log = np.log(latest_close / previous_close) # Log Return
        except (ValueError, ZeroDivisionError) as e:
            logger.error(f"Erreur calcul log return pour {symbol_market}: {e}. Prix: {previous_close} -> {latest_close}. Mise à jour sautée.")
            return

        # Assurer que l'actif existe dans le state
        current_ratings = self.get_ratings(symbol_market) # This also initializes if needed
        r_h = current_ratings.get('R_H', 0.0)
        r_a = current_ratings.get('R_A', 0.0)

        # Apply decay *before* adding new magnitude for smoother transition
        r_h *= self._decay_factor
        r_a *= self._decay_factor

        # Calculate weighted magnitude based on log return
        m = self.calculate_movement_magnitude(price_change_log)

        # Mettre à jour les ratings
        # Δp > 0  => renforce R_H (+λm), affaiblit R_A (-γm)
        # Δp < 0  => renforce R_A (+λm), affaiblit R_H (-γm)
        # Δp = 0  => m=0, pas de changement
        significance_threshold = 1e-9  # Minimum log return to consider a movement

        r_h_new, r_a_new = r_h, r_a # Start with decayed values

        if price_change_log > significance_threshold:
            # Mouvement Haussier
            r_h_new = r_h + self.LAMBDA * m
            r_a_new = max(0.0, r_a - self.GAMMA * m)  # Prevent R_A from becoming negative
        elif price_change_log < -significance_threshold:
            # Mouvement Baissier
            r_a_new = r_a + self.LAMBDA * m
            r_h_new = max(0.0, r_h - self.GAMMA * m)  # Prevent R_H from becoming negative
        else:
            # Mouvement negligible
            pass # Ratings remain the decayed values

        # Normalize the updated ratings
        r_h_new, r_a_new = self.normalize_ratings(r_h_new, r_a_new)

        # Recalculate confidence score based on the new ratings
        confidence = self.calculate_confidence_score(r_h_new, r_a_new)

        # Apply the update (avoid NaNs)
        self._ratings_state[symbol_market]['R_H'] = float(r_h_new) if np.isfinite(r_h_new) else 0.0
        self._ratings_state[symbol_market]['R_A'] = float(r_a_new) if np.isfinite(r_a_new) else 0.0

        # Add to history
        history_entry = {
            'timestamp': timestamp,
            'R_H': self._ratings_state[symbol_market]['R_H'], # Use the potentially updated values
            'R_A': self._ratings_state[symbol_market]['R_A'],
            'price': float(latest_close) if np.isfinite(latest_close) else np.nan,
            'confidence': float(confidence) if np.isfinite(confidence) else 0.0
        }

        if symbol_market not in self._ratings_history:
            self._ratings_history[symbol_market] = []

        self._ratings_history[symbol_market].append(history_entry)

        # Limit the history size
        if len(self._ratings_history[symbol_market]) > self._max_history_per_asset:
            self._ratings_history[symbol_market] = self._ratings_history[symbol_market][-(self._max_history_per_asset):]

        logger.debug(f"Ratings {symbol_market} updated: R_H={self._ratings_state[symbol_market]['R_H']:.4f}, R_A={self._ratings_state[symbol_market]['R_A']:.4f}, Confidence={confidence:.4f} (Mvt Log={price_change_log:.4f}, Magnitude={m:.4f})")

        # The save will happen externally (e.g., in main_trader)


# ===== TA CLASSIQUES (TALIB) & NEW TRINARY INPUTS =====

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> pd.Series:
        """Calculates Average True Range (ATR) using TALib."""
        if not (isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series)):
            logger.error("Input prices must be pandas Series for ATR calculation.")
            return pd.Series(dtype=float, index=close.index) # Return empty series with index on error

        if high.empty or low.empty or close.empty or not (len(high) == len(low) == len(close)):
            # logger.warning("Input price series for ATR calculation are empty or unequal length.") # Too verbose during backtest start
            return pd.Series(dtype=float, index=close.index) # Return empty series with index

        # Ensure data is float type expected by TA-Lib
        high_p, low_p, close_p = high.astype(float), low.astype(float), close.astype(float)

        try:
            # TA-Lib returns np.nan for the first timeperiod-1 values
            atr_values = talib.ATR(high_p.to_numpy(), low_p.to_numpy(), close_p.to_numpy(), timeperiod=timeperiod)
            return pd.Series(atr_values, index=high.index)
        except Exception as e:
            logger.error(f"Error calculating ATR with TALib: {e}", exc_info=True)
            return pd.Series(dtype=float, index=high.index) # Return empty series with index on error

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> pd.Series:
        """Calculates Average Directional Index (ADX) using TALib."""
        if not (isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series)):
            logger.error("Input prices must be pandas Series for ADX calculation.")
            return pd.Series(dtype=float, index=close.index) # Return empty series with index on error

        if high.empty or low.empty or close.empty or not (len(high) == len(low) == len(close)):
            return pd.Series(dtype=float, index=close.index) # Return empty series with index

        # Ensure data is float type expected by TA-Lib
        high_p, low_p, close_p = high.astype(float), low.astype(float), close.astype(float)

        try:
            adx_values = talib.ADX(high_p.to_numpy(), low_p.to_numpy(), close_p.to_numpy(), timeperiod=timeperiod)
            return pd.Series(adx_values, index=high.index)
        except Exception as e:
            logger.error(f"Error calculating ADX with TALib: {e}", exc_info=True)
            return pd.Series(dtype=float, index=high.index) # Return empty series with index on error


    def calculate_bb_width(self, close: pd.Series, timeperiod: int = 20, nbdevup: int = 2, nbdevdn: int = 2) -> pd.Series:
        """Calculates Bollinger Bands Width using TALib."""
        if not isinstance(close, pd.Series):
            logger.error(f"Input 'close' must be a pandas Series for BB_WIDTH calculation. Received type: {type(close)}. Returning empty Series.")
            return pd.Series(dtype=float) # Return empty series without trying to access .index
        if close.empty:
            logger.warning("Input 'close' series is empty for BB_WIDTH calculation. Returning empty Series.")
            return pd.Series(dtype=float, index=close.index) # Return empty series with original index if it's a Series

        # Ensure data is float type expected by TA-Lib
        close_p = close.astype(float)

        try:
            upperband, middleband, lowerband = talib.BBANDS(close_p.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0)
            # Avoid division by zero in middleband, especially at the start of the series
            bb_width = pd.Series(dtype=float, index=close.index)
            non_zero_middleband_idx = middleband > 1e-9 # Use a small epsilon
            if np.any(non_zero_middleband_idx):
                 bb_width[non_zero_middleband_idx] = ((upperband[non_zero_middleband_idx] - lowerband[non_zero_middleband_idx]) / middleband[non_zero_middleband_idx]) * 100 # As a percentage
            # Fill NaNs resulting from early bars or zero middleband
            return bb_width.fillna(0) # Fill NaNs with 0 or use ffill/bfill depending on desired behavior
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands Width with TALib: {e}", exc_info=True)
            return pd.Series(dtype=float, index=close.index) # Return empty series with index on error


    def calculate_all_ta_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule un ensemble d'indicateurs techniques classiques (et ceux nécessaires pour le Trinary System)
        sur les données OHLCV.
        Retourne un DataFrame avec les indicateurs en colonnes.
        """
        # Assurer que les colonnes OHLCV existent et sont du bon type
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                # logger.error(f"Colonne manquante '{col}' dans les données pour le calcul des indicateurs TA.") # Too verbose
                return pd.DataFrame(index=data.index) # Return an empty DF with index on critical error

        indicators_df = pd.DataFrame(index=data.index)

        # Ensure core OHLCV columns are numeric for TA-Lib
        try:
            # Create copies to avoid modifying the input DataFrame in place
            open_p = pd.to_numeric(data['open'], errors='coerce')
            high_p = pd.to_numeric(data['high'], errors='coerce')
            low_p = pd.to_numeric(data['low'], errors='coerce')
            close_p = pd.to_numeric(data['close'], errors='coerce')
            volume_p = pd.to_numeric(data['volume'], errors='coerce')
            
            # Drop rows where essential price columns are NaN after coercion
            data_cleaned = pd.DataFrame({
                'open': open_p, 'high': high_p, 'low': low_p, 'close': close_p, 'volume': volume_p
            }).dropna(subset=['open', 'high', 'low', 'close'])
            logger.debug(f"Data cleaned for TA calculation. Original length: {len(data)}, Cleaned length: {len(data_cleaned)}. Original index type: {data.index.dtype}, Cleaned index type: {data_cleaned.index.dtype}")
            
            # Use the index from the cleaned data for indicators_df
            indicators_df = indicators_df.reindex(data_cleaned.index)

            if data_cleaned.empty:
                logger.warning("Données vides ou converties en NaN après nettoyage pour calcul TA.")
                return pd.DataFrame(index=data.index) # Return empty DF with original index

        except ValueError as e:
            logger.error(f"Impossible de convertir les colonnes OHLCV en numérique pour TA-Lib: {e}")
            return pd.DataFrame(index=data.index) # Return empty DF with original index

        # Use .get for parameters with default values from indicator_config dict
        # Ensure indicator_config is a dict and handle potential missing sections if needed
        if not isinstance(self._indicator_config, dict):
            logger.error("indicator_config est invalide ou manquant. Utilisation des paramètres par défaut.")
            self._indicator_config = {} # Use empty dict if input is not valid


        # SMA
        sma_config = self._indicator_config.get('sma', {})
        if sma_config.get('enabled', False):
            sma_periods = sma_config.get('periods', [10, 50])
            for period in sma_periods:
                try:
                    indicators_df[f'SMA_{period}'] = talib.SMA(data_cleaned['close'], timeperiod=int(period))
                except Exception as e:
                    logger.error(f"Erreur calcul SMA_{period}: {e}")

        # RSI
        rsi_config = self._indicator_config.get('rsi', {})
        if rsi_config.get('enabled', False):
            rsi_period = rsi_config.get('period', 14)
            try:
                indicators_df['RSI'] = talib.RSI(data_cleaned['close'], timeperiod=int(rsi_period))
            except Exception as e:
                logger.error(f"Erreur calcul RSI: {e}")

        # MACD
        macd_config = self._indicator_config.get('macd', {})
        if macd_config.get('enabled', False):
            macd_fast = macd_config.get('fastperiod', 12)
            macd_slow = macd_config.get('slowperiod', 26)
            macd_signal = macd_config.get('signalperiod', 9)
            try:
                macd, macdsignal, macdhist = talib.MACD(data_cleaned['close'],
                                                        fastperiod=int(macd_fast),
                                                        slowperiod=int(macd_slow),
                                                        signalperiod=int(macd_signal))
                indicators_df['MACD'] = macd
                indicators_df['MACD_signal'] = macdsignal
                indicators_df['MACD_hist'] = macdhist
            except Exception as e:
                logger.error(f"Erreur calcul MACD: {e}")

        # ATR (Needed for Trinary IFT & RDVU, and for Regime)
        atr_config = self._indicator_config.get('atr', {})
        # Default period for ATR is 14, check enabled from config or assume needed if ATR is a dependency
        atr_period = atr_config.get('period', 14) # Standard ATR period
        try:
            # Use the helper function defined above
            indicators_df['ATR'] = self.calculate_atr(data_cleaned['high'], data_cleaned['low'], data_cleaned['close'], timeperiod=int(atr_period))
        except Exception as e:
            logger.error(f"Erreur calcul ATR: {e}")


        # SMA of ATR (Needed for Trinary RDVU) - Uses different periods
        # Note: The snippet uses keys like 'trinary_atr_period_volatility' for the base ATR period
        # and 'trinary_atr_sma_period' for the SMA period. Let's align config keys.
        trinary_volatility_config = self._indicator_config.get('trinary_volatility', {}) # Assuming a new section or keys under indicators
        atr_base_period_for_sma = trinary_volatility_config.get('atr_period', 14) # Default to standard ATR period if not specified
        atr_sma_period = trinary_volatility_config.get('atr_sma_period', 50)

        # Calculate ATR specifically for the SMA_ATR calculation if a different base period is needed
        # Or just use the 'ATR' column calculated above if atr_base_period_for_sma is the same as atr_period
        atr_series_for_sma = indicators_df.get('ATR') # Start with the standard ATR column

        if atr_base_period_for_sma != atr_period or atr_series_for_sma is None:
             # Recalculate ATR if a different base period is specified for SMA_ATR, or if ATR calc above failed
             logger.debug(f"Recalculating ATR with period {atr_base_period_for_sma} for SMA_ATR calculation.")
             try:
                 atr_series_for_sma = self.calculate_atr(data_cleaned['high'], data_cleaned['low'], data_cleaned['close'], timeperiod=int(atr_base_period_for_sma))
             except Exception as e:
                 logger.error(f"Erreur calcul base ATR ({atr_base_period_for_sma}) pour SMA_ATR: {e}")
                 atr_series_for_sma = pd.Series(dtype=float, index=data_cleaned.index) # Empty series on error


        # Calculate SMA on the ATR series
        indicators_df['SMA_ATR'] = pd.Series(dtype=float, index=data_cleaned.index) # Initialize column
        if atr_series_for_sma is not None and not atr_series_for_sma.empty:
            valid_atr = atr_series_for_sma.dropna()
            if len(valid_atr) >= atr_sma_period:
                try:
                    sma_atr_values = talib.SMA(valid_atr.to_numpy(), timeperiod=int(atr_sma_period))
                    # Reindex the result to match the original index
                    sma_atr_series = pd.Series(sma_atr_values, index=valid_atr.index)
                    indicators_df['SMA_ATR'] = sma_atr_series.reindex(indicators_df.index).fillna(method='ffill') # Use ffill to propagate values
                    # indicators_df['SMA_ATR'].fillna(method='bfill', inplace=True) # Optional: bfill start if ffill leaves NaNs
                except Exception as e:
                    logger.error(f"Erreur calcul SMA_ATR ({atr_sma_period}) with TALib: {e}", exc_info=True)
            else:
                pass
                 # logger.warning(f"Pas assez de données ATR valides ({len(valid_atr)} < {atr_sma_period}) pour calculer SMA_ATR.") # Too verbose


        # BB_WIDTH (Needed for Trinary RDVU)
        bbw_config = self._indicator_config.get('bb_width', {}) # Use .get first
        if bbw_config.get('enabled', True): # Enabled by default for regime and Trinary
            bbw_period = bbw_config.get('period', 20)
            bbw_nbdev = bbw_config.get('nbdev', 2)
            try:
                # Use the helper function defined above
                indicators_df['BB_WIDTH'] = self.calculate_bb_width(data_cleaned['close'], timeperiod=int(bbw_period), nbdevup=float(bbw_nbdev), nbdevdn=float(bbw_nbdev))
            except Exception as e:
                logger.error(f"Erreur calcul BB_WIDTH: {e}")

        # ADX (Needed for Regime)
        adx_config = self._indicator_config.get('adx', {})
        if adx_config.get('enabled', True): # Enabled by default for regime
            adx_period = adx_config.get('period', 14)
            try:
                # Use the helper function defined above
                indicators_df['ADX'] = self.calculate_adx(data_cleaned['high'], data_cleaned['low'], data_cleaned['close'], timeperiod=int(adx_period))
            except Exception as e:
                logger.error(f"Erreur calcul ADX: {e}")


        # Add other indicators here according to configuration...
        # EMA, Stoch, etc.

        logger.debug(f"Indicateurs TA calculés: {indicators_df.columns.tolist()} pour {len(data_cleaned)} barres.")
        # Use original index for the final DataFrame before dropping NaNs
        final_indicators_df = indicators_df.dropna(how='all').reindex(data.index) # Drop lines where all calculated indicators are NaN (start of series)
        logger.debug(f"Final TA indicators DataFrame reindexed to original data index. Final length: {len(final_indicators_df)}. Final index type: {final_indicators_df.index.dtype}")
        return final_indicators_df


# ===== TRINARY SYSTEM FUNCTIONS (Section I & II) =====

# Section I: R-System and IFT

    def calculate_r_equilibrium(self, r_h: float, r_a: float) -> float:
        """Calculates the equilibrium rating R_equilibrium = R_H - R_A."""
        # Ensure inputs are finite
        if not np.isfinite(r_h): r_h = 0.0
        if not np.isfinite(r_a): r_a = 0.0
        return r_h - r_a

    def calculate_r_growth(self, r_equilibrium: float) -> float:
        """Calculates the growth rating R_growth = |R_equilibrium|."""
        # Ensure input is finite
        if not np.isfinite(r_equilibrium): r_equilibrium = 0.0
        return abs(r_equilibrium)

    def calculate_r_cycle(self, r_equilibrium: float, market_phase: float) -> float:
        """
        Calculates the cyclic rating R_cycle.
        Args:
            r_equilibrium: The equilibrium rating (R_H - R_A).
            market_phase: The current market phase, expected in the range [0, 2].
                           (Derived from Pi-Rating Confidence Score + 1).
        Returns:
            The cyclic rating.
        """
        # Ensure inputs are finite
        if not np.isfinite(r_equilibrium): r_equilibrium = 0.0
        if not np.isfinite(market_phase): market_phase = 0.0

        # Clamp market_phase to [0, 2] to avoid issues with sin function
        market_phase = np.clip(market_phase, 0.0, 2.0)

        # Calculate the cyclic component: maps [0, 2] to [0, 1, 0] shape via sin(PI * phase / 2)
        comp_cycle = math.sin(self.PI * market_phase / 2.0)

        # R_cycle = R_equilibrium × sin(π × market_phase / 2)
        # Note: This formulation might imply R_cycle oscillates around 0,
        # or it might be intended as a component influencing R_unified.
        # Based on the formula and R_equilibrium as a factor, it seems to scale the equilibrium signal cyclically.
        return r_equilibrium * comp_cycle


    def calculate_r_unified(self, r_equilibrium: float, r_growth: float, r_cycle: float) -> float:
        """
        Calculates the unified rating R_unified based on equilibrium, growth, and cyclic components.
        Formula: R_unified = R_equilibrium + R_growth + R_cycle
        """
        # Ensure inputs are finite
        if not np.isfinite(r_equilibrium): r_equilibrium = 0.0
        if not np.isfinite(r_growth): r_growth = 0.0
        if not np.isfinite(r_cycle): r_cycle = 0.0

        # Simple sum as per the formula
        return r_equilibrium + r_growth + r_cycle


    def calculate_ift(self, r_h: float, r_a: float, market_momentum: float, stability_factor: float) -> Tuple[float, float]:
        """
        Calculates the Inverse Fisher Transform (IFT) based ratings (IFT_bull, IFT_bear).
        Args:
            r_h: Haussier rating (from Pi-Ratings or R-System).
            r_a: Baissier rating (from Pi-Ratings or R-System).
            market_momentum: Market momentum indicator (e.g., scaled RSI [-1, 1]).
            stability_factor: Market stability indicator (e.g., inverse of normalized ATR).
        Returns:
            Tuple (IFT_bull, IFT_bear).
        """
        # Ensure inputs are finite
        if not np.isfinite(r_h): r_h = 0.0
        if not np.isfinite(r_a): r_a = 0.0
        if not np.isfinite(market_momentum): market_momentum = 0.0
        if not np.isfinite(stability_factor): stability_factor = 1.0 # Default stability to 1.0 if NaN

        # Avoid extreme values in stability_factor which might cause overflow in exp
        stability_factor = np.clip(stability_factor, 1e-3, 1e3) # Example clipping range

        # Midpoint based on R_H, R_A, and market momentum
        # This interpretation assumes market_momentum adds bias to R_H/R_A difference.
        # A common IFT application is on a single bounded oscillator [-1, 1].
        # Here, we need to adapt. Let's try combining R_H/R_A with momentum into a single value [-1, 1] proxy.
        # The R-System Confidence Score (scaled R_H - R_A) is a good proxy for [-1, 1].
        # Let's use the scaled confidence score as the primary input to IFT, biased by market_momentum.
        confidence_score = self.calculate_confidence_score(r_h, r_a) # Range [-1, 1]

        # Combine confidence and momentum. Weighted average? Simple sum? Let's do a weighted sum.
        # Assume confidence is more important, but momentum provides edge. Weights could be configurable.
        # For now, let's use fixed weights or just a slightly biased confidence.
        # Using the confidence score itself as the input for IFT, perhaps slightly adjusted by momentum.
        # A typical IFT is `1 / (1 + exp(-2 * x)) * 2 - 1` where x is bounded [-1, 1].
        # Our confidence score is already bounded [-1, 1]. Let's use that.

        # Applying IFT to the confidence score directly
        # Fisher Transform: 0.5 * log((1+x)/(1-x))
        # Inverse Fisher Transform: (exp(2y) - 1) / (exp(2y) + 1), where y is Fisher Transform output
        # Let's use a common IFT implementation often seen: IFT = (exp(2 * (confidence_score + momentum_bias)) - 1) / (exp(2 * (confidence_score + momentum_bias)) + 1)
        # But our confidence score is already [-1, 1]. Let's apply IFT to it.

        # The formula implies IFT(R_H - R_A) influenced by market_momentum and stability.
        # Let's interpret it as IFT applied to a combined signal influenced by all factors.
        # Signal = (R_H - R_A) * f(momentum, stability) ? Or apply IFT on a derived signal?
        # Let's reinterpret based on the structure IFT_bull and IFT_bear output.
        # IFT often outputs values that swing between 0 and 1 or -1 and 1.
        # IFT_bull and IFT_bear might be 1/(1+exp(-x)) and 1/(1+exp(x)) outputs, or similar.

        # Let's assume the formula refers to applying a sigmoid-like transform:
        # The formula looks more like a variation combining terms directly rather than standard IFT on a single variable.
        # Let's use the formulation: IFT_bull = f(R_H, momentum, stability), IFT_bear = f(R_A, momentum, stability)
        # Maybe it's meant to be IFT applied to R_H and R_A scaled by other factors?

        # Reverting to a common IFT pattern: Apply IFT to a single score that combines the inputs.
        # Combined Score = w1 * (R_H - R_A) + w2 * market_momentum + w3 * stability_factor?
        # This doesn't seem right as inputs have different scales.
        # Let's consider the formula implied by the output names IFT_bull/bear.
        # IFT_bull = sigmoid(signal), IFT_bear = sigmoid(-signal) or 1 - IFT_bull.
        # The 'signal' could be a combination of R_H, R_A, momentum, stability.

        # Let's go back to the simplest interpretation of the output names IFT_bull/bear being related to the sigmoid function:
        # sigmoid(x) = 1 / (1 + exp(-x))
        # IFT_bull = sigmoid(some_signal_x)
        # IFT_bear = sigmoid(-some_signal_x) or sigmoid(another_signal_y)

        # Let's assume the signal input 'x' to the sigmoid as:
        # x = (r_h - r_a) + market_momentum * stability_factor # Example combination
        # Need to ensure 'x' doesn't cause exp() to overflow/underflow. Clip x.
        x = (r_h - r_a) + market_momentum * stability_factor

        # Apply standard sigmoid transformation
        # sigmoid(x) = 1 / (1 + exp(-x))
        # To avoid overflow/underflow with exp(-x), handle large/small x values.
        # If x is very large positive, exp(-x) is near 0, sigmoid is near 1.
        # If x is very large negative, exp(-x) is very large, sigmoid is near 0.
        # If x = 0, sigmoid is 0.5.
        # Let's clip x before exp. Common range for sigmoid input is -10 to +10.
        x = np.clip(x, -10.0, 10.0)

        # IFT_bull could be 1 / (1 + exp(-x))
        # IFT_bear could be 1 / (1 + exp(x)) which is 1 - IFT_bull.
        # So, IFT_bull + IFT_bear = 1. This is a common IFT pair.

        ift_bull = 1.0 / (1.0 + math.exp(-x))
        ift_bear = 1.0 / (1.0 + math.exp(x)) # This is 1 - ift_bull

        # Ensure outputs are finite floats
        ift_bull = float(ift_bull) if np.isfinite(ift_bull) else 0.5
        ift_bear = float(ift_bear) if np.isfinite(ift_bear) else 0.5 # Should also be 1 - ift_bull if ift_bull is finite

        return ift_bull, ift_bear


# Section II: Volatility System

    def calculate_rdvu(self, atr_current: float, atr_average: float, 
                           volatility_cycle_position: float, # Expected range [0, 2*pi] (from BBW %ile * 2 * pi)
                           base_stability_index: float,    # Expected range 0-1 (from abs(sentiment))
                           pi_factor: float = None, # pi_factor is kept for consistency, but direct usage in VCC changes
                           cycle_amplitude: Optional[float] = None) -> float:
        """
        Calculates the R_DVU (Dynamic Volatility Understanding) component of the Trinary System.

        Args:
            atr_current (float): Current ATR value.
            atr_average (float): Smoothed average of ATR (e.g., SMA of ATR).
            volatility_cycle_position (float): Derived from BB_WIDTH percentile rank, mapped to [0, 2*pi].
            base_stability_index (float): Derived from absolute sentiment score, clipped to 0-1.
            pi_factor (float, optional): Mathematical constant Pi. Defaults to np.pi. 
                                         Note: If volatility_cycle_position is already scaled by Pi, 
                                         pi_factor's direct use in VCC's cosine term is adjusted.
            cycle_amplitude (float, optional): Amplitude for the volatility cycle. 
                                             Defaults to atr_current if None.

        Returns:
            float: The R_DVU value.
        """
        if pi_factor is None:
            pi_factor = self.PI # Use instance PI

        if cycle_amplitude is None:
            cycle_amplitude = atr_current

        if atr_average is None or atr_average == 0:
            logger.warning("atr_average is None or zero in calculate_rdvu. RDVU might be unstable. Returning 0.")
            return 0.0
        if atr_current is None:
            logger.warning("atr_current is None in calculate_rdvu. RDVU might be unstable. Returning 0.")
            return 0.0
        if cycle_amplitude is None or cycle_amplitude < 0: # atr_current can't be negative, but good to check
            logger.warning(f"cycle_amplitude '{cycle_amplitude}' is invalid in calculate_rdvu. Returning 0.")
            return 0.0

        # Ensure volatility_cycle_position is within its expected range [0, 2*pi]
        volatility_cycle_position = np.clip(volatility_cycle_position, 0, 2 * self.PI) # Use PI directly

        # Ensure base_stability_index is within 0-1
        base_stability_index = np.clip(base_stability_index, 0, 1)

        try:
            # Volatility Ratio Component (VRC)
            # Measures current volatility relative to its recent average.
            # Added a small epsilon to prevent division by zero if atr_average is extremely small (though checked above).
            vrc = atr_current / (atr_average + 1e-9) 

            # Volatility Cycle Component (VCC)
            # Models cyclicality in volatility. Uses volatility_cycle_position (now [0, 2*pi]) as input to a cosine wave.
            # The mapping percentile_rank * 2 * pi for volatility_cycle_position means:
            # 0 (low BBW %ile) -> cos(0) = 1
            # pi (mid BBW %ile) -> cos(pi) = -1
            # 2pi (high BBW %ile) -> cos(2pi) = 1
            # This creates a U-shape: higher VCC at low and high BBW %iles, lower VCC at mid %iles.
            # The cycle_amplitude (atr_current) scales this component.
            vcc = cycle_amplitude * np.cos(volatility_cycle_position) # Adjusted: volatility_cycle_position is now the angle

            # Stability-Adjusted Volatility Index (SAVI)
            # Combines VRC and VCC, then adjusts by base_stability_index.
            # Higher stability (base_stability_index closer to 1) dampens the raw volatility measure.
            # (1 - base_stability_index) means: 1 for no stability (index=0), 0 for full stability (index=1)
            savi = (vrc + vcc) * (1 - base_stability_index) 

            # R_DVU: Normalize SAVI to a more bounded range, e.g., using a sigmoid or tanh.
            # For now, let's return SAVI directly. Normalization can be part of feature engineering if needed.
            # Or, if SAVI can be negative and a positive-only or specific range is desired:
            # r_dvu = np.tanh(savi) # Example: maps to [-1, 1]
            r_dvu = savi # Returning raw SAVI for now

            if not np.isfinite(r_dvu):
                logger.warning(f"R_DVU calculated as non-finite ({r_dvu}) with inputs: atr_current={atr_current}, atr_average={atr_average}, vol_cycle_pos={volatility_cycle_position}, base_stability={base_stability_index}. Returning 0.")
                return 0.0
            return float(r_dvu)
        except Exception as e:
            logger.error(f"Error in calculate_rdvu: {e}", exc_info=True)
            return 0.0

    def classify_market_regime_trinary(self, rdvu: float) -> str:
        """
        Classifies the market regime based on the R_DVU value.
        This is a conceptual mapping, thresholds need to be determined empirically.
        """
        # Ensure input is finite
        if not np.isfinite(rdvu): rdvu = 0.0

        # Example thresholds (NEED EMPIRICAL TUNING)
        # Low RDVU might mean low volatility relative to average, potential STABLE/RANGE
        # Medium RDVU (T1 to T2): CYCLICAL (Volatility has a pattern, potentially oscillating)
        # High RDVU (> T2): TRENDING (High directional volatility)
        # Very High RDVU (> T3): CHAOTIC (Extreme, unpredictable volatility)

        # These thresholds need to be determined from analyzing the distribution of RDVU over historical data.
        # Placeholder thresholds:
        regime_thresholds = self._indicator_config.get('trinary_volatility', {}).get('regime_thresholds_rdvu', '0.5, 1.5, 2.5')
        try:
            t1, t2, t3 = [float(t.strip()) for t in regime_thresholds.split(',')]
        except Exception:
            logger.warning(f"Invalid RDVU regime thresholds config '{regime_thresholds}'. Using defaults (0.5, 1.5, 2.5).")
            t1, t2, t3 = 0.5, 1.5, 2.5

        if rdvu < t1:
            return "STABLE" # Low volatility, ranging
        elif t1 <= rdvu < t2:
            return "CYCLICAL" # Volatility has some pattern/oscillation
        elif t2 <= rdvu < t3:
            return "TRENDING" # Higher volatility, potentially directional movement
        else: # rdvu >= t3
            return "CHAOTIC" # Very high volatility, possibly unpredictable


# ===== HELPER FUNCTIONS TO DERIVE TRINARY INPUTS =====
# These functions take outputs of other standard/Pi indicators and transform them
# into the specific inputs needed by the core Trinary calculation functions.

    def derive_market_phase(self, confidence_score: float) -> float:
        """
        Derives the market phase from the Pi-Rating Confidence Score.
        Maps confidence_score [-1, 1] to market_phase [0, 2].
        """
        if not np.isfinite(confidence_score): confidence_score = 0.0
        # Formula: phase = confidence_score + 1
        # -1 -> 0, 0 -> 1, 1 -> 2
        return np.clip(confidence_score + 1.0, 0.0, 2.0)


    def derive_market_momentum(self, rsi_value: Optional[float]) -> float:
        """
        Derives market momentum from RSI.
        Maps RSI [0, 100] to momentum [-1, 1].
        """
        # Default RSI is 50 (neutral) if input is NaN or None
        rsi_value = float(rsi_value) if np.isfinite(rsi_value) else 50.0
        # Clamp RSI to valid range just in case
        rsi_value = np.clip(rsi_value, 0.0, 100.0)
        # Formula: momentum = (RSI - 50) / 50
        # 0 -> -1, 50 -> 0, 100 -> 1
        return (rsi_value - 50.0) / 50.0


    def derive_stability_factor(self, close_price: float, atr_value: Optional[float]) -> float:
        """
        Derives the stability factor from Close Price and ATR (inverse of Normalized ATR).
        stability_factor = Close Price / ATR
        """
        # Ensure inputs are finite and positive for price/ATR
        close_price = float(close_price) if np.isfinite(close_price) and close_price > 0 else 1e-9
        atr_value = float(atr_value) if np.isfinite(atr_value) and atr_value > 0 else 1e-9 # Use epsilon for zero/near-zero ATR

        # Formula: stability_factor = Close Price / ATR
        return close_price / atr_value


    def derive_volatility_cycle_position(self, bbw_percentile_rank: Optional[float]) -> float:
        """
        Derives volatility cycle position from BB_WIDTH percentile rank.
        Maps percentile_rank [0, 1] to position [0, 2*pi].
        """
        # Default percentile rank is 0.5 (median) if input is NaN or None
        percentile_rank = float(bbw_percentile_rank) if np.isfinite(bbw_percentile_rank) else 0.5
        # Clamp percentile rank to valid range
        percentile_rank = np.clip(percentile_rank, 0.0, 1.0)
        # Formula: position = percentile_rank * 2 * pi
        # 0 -> 0, 0.5 -> pi, 1 -> 2pi
        return percentile_rank * 2.0 * self.PI


    def derive_cycle_amplitude(self, atr_current: Optional[float]) -> float:
        """
        Derives the volatility cycle amplitude, which is the current ATR value.
        """
        # Default ATR is 0 if input is NaN or None
        atr_current = float(atr_current) if np.isfinite(atr_current) else 0.0
        # Ensure amplitude is non-negative
        return max(0.0, atr_current)


    def derive_base_stability_index(self, sentiment_score: Optional[float]) -> float:
        """
        Derives the base stability index from the sentiment score.
        Uses abs(sentiment_score) clipped to [0, 1].
        """
        # Default sentiment score is 0 (neutral) if input is NaN or None
        sentiment_score = float(sentiment_score) if np.isfinite(sentiment_score) else 0.0
        # Formula: base_stability_index = abs(sentiment_score)
        # -1 -> 1, 0 -> 0, 1 -> 1
        # Clamp to [0, 1] just in case abs() results in slightly > 1 due to floating point
        return np.clip(abs(sentiment_score), 0.0, 1.0)

    def derive_pi_factor(self) -> float:
        """Returns the mathematical constant Pi."""
        return self.PI


# ===== WRAPPER FOR CONSISTENT API =====
    def initialize(self, config: ConfigParser):
        """
        Wrapper for initialize_indicators for consistency.

        Args:
            config: Objet ConfigParser contenant la configuration du système
        """
        self._initialize_indicators_internal()

# ===== MARKET REGIME CLASSIFICATION (using standard TA) =====

    def classify_market_regime(self, indicators_df: pd.DataFrame, config_manager: Any) -> pd.Series:
        """
        Classifies the market regime based on ADX, ATR, and Bollinger Bands Width.

        Args:
            indicators_df: DataFrame containing 'ADX', 'ATR', 'BB_WIDTH' columns.
            config: ConfigParser object to load thresholds.

        Returns:
            A pandas Series with regime classifications (e.g., "Trending_Volatile", "Ranging_Calm").
        """
        if not all(col in indicators_df.columns for col in ['ADX', 'ATR', 'BB_WIDTH']):
            # logger.warning("Missing one or more required columns (ADX, ATR, BB_WIDTH) for regime classification.") # Too verbose
            return pd.Series(index=indicators_df.index, dtype=str).fillna("Indeterminate")

        # Default thresholds (can be overridden by config)
        adx_trend_threshold = 25
        adx_range_threshold = 20
        atr_volatility_high_factor = 1.5 # ATR is 1.5x its 50-period SMA
        atr_volatility_low_factor = 0.75 # ATR is 0.75x its 50-period SMA
        bbw_expansion_threshold_percentile = 0.75 # Default percentile
        bbw_contraction_threshold_percentile = 0.25 # Default percentile
        atr_smoothing_period = 50

        # Calculate initial percentile-based BBW thresholds before potentially overriding with fixed values from config
        # Ensure BB_WIDTH column exists and is not all NaN before calculating quantiles
        bbw_series = indicators_df.get('BB_WIDTH')
        if bbw_series is not None and not bbw_series.dropna().empty:
            bbw_expansion_threshold = bbw_series.quantile(bbw_expansion_threshold_percentile)
            bbw_contraction_threshold = bbw_series.quantile(bbw_contraction_threshold_percentile)
        else:
            # Fallback if BB_WIDTH is missing or all NaN
            bbw_expansion_threshold = float('inf')
            bbw_contraction_threshold = float('-inf')
            logger.warning("BB_WIDTH data missing or all NaN. Percentile BBW thresholds cannot be calculated.")


        config_regime = config_manager # Use the passed config_manager directly

        adx_trend_threshold = config_manager.getfloat('regime_thresholds', 'adx_trend', fallback=adx_trend_threshold)
        adx_range_threshold = config_manager.getfloat('regime_thresholds', 'adx_range', fallback=adx_range_threshold)
        atr_volatility_high_factor = config_manager.getfloat('regime_thresholds', 'atr_volatility_high_factor', fallback=atr_volatility_high_factor)
        atr_volatility_low_factor = config_manager.getfloat('regime_thresholds', 'atr_volatility_low_factor', fallback=atr_volatility_low_factor)
        atr_smoothing_period = config_manager.getint('regime_thresholds', 'atr_smoothing_period', fallback=atr_smoothing_period)

        # Allow config to specify percentiles or fixed values for BBW thresholds
        # Use fallback values if config options are missing
        bbw_expansion_threshold = config_manager.getfloat('regime_thresholds', 'bbw_expansion_fixed', fallback=2.0)
        bbw_contraction_threshold = config_manager.getfloat('regime_thresholds', 'bbw_contraction_fixed', fallback=0.5)
        
        # Recalculate percentile-based BBW thresholds if fixed values are not used or explicitly requested
        if config_manager.get('regime_thresholds', 'bbw_expansion_fixed') is None and bbw_series is not None and not bbw_series.dropna().empty:
            bbw_expansion_threshold_percentile = config_manager.getfloat('regime_thresholds', 'bbw_expansion_percentile', fallback=0.75)
            bbw_expansion_threshold = bbw_series.quantile(bbw_expansion_threshold_percentile)
        
        if config_manager.get('regime_thresholds', 'bbw_contraction_fixed') is None and bbw_series is not None and not bbw_series.dropna().empty:
            bbw_contraction_threshold_percentile = config_manager.getfloat('regime_thresholds', 'bbw_contraction_percentile', fallback=0.25)
            bbw_contraction_threshold = bbw_series.quantile(bbw_contraction_threshold_percentile)


        regimes = pd.Series(index=indicators_df.index, dtype=str).fillna("Indeterminate")

        # Calculate dynamic ATR thresholds
        # Ensure ATR is not all NaN before calculating SMA, handle short series
        atr_series = indicators_df.get('ATR')
        valid_atr = atr_series.dropna() if atr_series is not None else pd.Series(dtype=float)

        if len(valid_atr) >= atr_smoothing_period:
            try:
                atr_sma = talib.SMA(valid_atr.to_numpy(), timeperiod=int(atr_smoothing_period))
                # Reindex the result to match the original index
                atr_sma_series = pd.Series(atr_sma, index=valid_atr.index)
                indicators_df['SMA_ATR'] = atr_sma_series.reindex(indicators_df.index).fillna(method='ffill') # Use ffill to propagate values
                # indicators_df['SMA_ATR'].fillna(method='bfill', inplace=True) # Optional: bfill start if ffill leaves NaNs
                atr_high_threshold = atr_sma_series * atr_volatility_high_factor
                atr_low_threshold = atr_sma_series * atr_volatility_low_factor
                # Ensure thresholds are not NaN/Inf if atr_sma_series calculation failed partially
                atr_high_threshold = atr_high_threshold.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                atr_low_threshold = atr_low_threshold.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            except Exception as e:
                logger.error(f"Error calculating ATR SMA for regime thresholds: {e}", exc_info=True)
                # Fallback to simple ATR thresholds or 0
                if atr_series is not None and not atr_series.dropna().empty:
                     mean_atr = atr_series.mean()
                     atr_high_threshold = pd.Series(mean_atr * atr_volatility_high_factor, index=indicators_df.index)
                     atr_low_threshold = pd.Series(mean_atr * atr_volatility_low_factor, index=indicators_df.index)
                else:
                     atr_high_threshold = pd.Series(0, index=indicators_df.index)
                     atr_low_threshold = pd.Series(0, index=indicators_df.index)
                logger.warning("All ATR is NaN or empty. ATR-based regime thresholds set to 0.")

        # Conditions
        adx_series = indicators_df.get('ADX', pd.Series(dtype=float, index=indicators_df.index).fillna(0)) # Default ADX to 0 if missing
        bbw_series = indicators_df.get('BB_WIDTH', pd.Series(dtype=float, index=indicators_df.index).fillna(0)) # Default BBW to 0 if missing

        is_trending = adx_series > adx_trend_threshold
        is_ranging = adx_series < adx_range_threshold

        # Compare current ATR/BBW to their dynamic thresholds (calculated above)
        # Reindex atr_series to match indicators_df.index to ensure alignment for comparisons
        atr_series_aligned = atr_series.reindex(indicators_df.index) if atr_series is not None else pd.Series(dtype=float, index=indicators_df.index)
        
        # Ensure thresholds are Series aligned with indicators_df.index
        atr_high_threshold_aligned = pd.Series(atr_high_threshold, index=indicators_df.index) if not isinstance(atr_high_threshold, pd.Series) else atr_high_threshold.reindex(indicators_df.index)
        atr_low_threshold_aligned = pd.Series(atr_low_threshold, index=indicators_df.index) if not isinstance(atr_low_threshold, pd.Series) else atr_low_threshold.reindex(indicators_df.index)

        is_volatile_atr = (atr_series_aligned.fillna(0) > atr_high_threshold_aligned.fillna(0))
        is_calm_atr = (atr_series_aligned.fillna(0) < atr_low_threshold_aligned.fillna(0))

        is_expanding_bbw = bbw_series > bbw_expansion_threshold
        is_contracting_bbw = bbw_series < bbw_contraction_threshold


        # --- Classification Logic ---
        # Order of assignment matters if conditions overlap; more specific first or use exclusive conditions.
        # Use a temporary copy or work on indices to avoid issues with inplace assignment changing subsequent conditions
        regimes_temp = pd.Series("Indeterminate", index=indicators_df.index, dtype=str)


        # Trending Strong (Volatile/Expanding)
        trending_volatile_cond = is_trending & (is_volatile_atr | is_expanding_bbw)
        regimes_temp[trending_volatile_cond] = "Trending_Volatile"

        # Trending Normal/Weak (Less Volatile)
        is_normal_volatility_atr = (~is_volatile_atr) & (~is_calm_atr) if atr_series is not None else pd.Series(False, index=indicators_df.index)
        is_normal_expansion_bbw = (~is_expanding_bbw) & (~is_contracting_bbw)
        trending_normal_cond = is_trending & (is_normal_volatility_atr | is_normal_expansion_bbw)
        regimes_temp[trending_normal_cond & (regimes_temp == "Indeterminate")] = "Trending_Normal"

        # Trending Calm/Contraction
        trending_calm_cond = is_trending & is_calm_atr & is_contracting_bbw
        regimes_temp[trending_calm_cond & (regimes_temp == "Indeterminate")] = "Trending_Calm_Contraction"


        # Ranging Volatile (Choppy)
        ranging_volatile_cond = is_ranging & (is_volatile_atr | is_expanding_bbw)
        regimes_temp[ranging_volatile_cond & (regimes_temp == "Indeterminate")] = "Ranging_Volatile"

        # Ranging Calm
        ranging_calm_cond = is_ranging & (is_calm_atr | is_contracting_bbw)
        regimes_temp[ranging_calm_cond & (regimes_temp == "Indeterminate")] = "Ranging_Calm"

        # Transitioning states for ADX between range and trend thresholds
        is_transitioning_adx = (adx_series >= adx_range_threshold) & (adx_series <= adx_trend_threshold)
        transition_volatile_cond = is_transitioning_adx & (is_volatile_atr | is_expanding_bbw)
        regimes_temp[transition_volatile_cond & (regimes_temp == "Indeterminate")] = "Transition_Volatile"

        transition_calm_cond = is_transitioning_adx & (is_calm_atr | is_contracting_bbw)
        regimes_temp[transition_calm_cond & (regimes_temp == "Indeterminate")] = "Transition_Calm"

        # Default any remaining Indeterminate based on primary ADX state if volatility is neutral
        # is_normal_volatility = (~is_volatile_atr) & (~is_calm_atr) & (~is_expanding_bbw) & (~is_contracting_bbw) # More strict neutral vol

        regimes_temp[is_trending & (regimes_temp == "Indeterminate")] = "Trending_Normal" # Default for trend if no strong vol signal
        regimes_temp[is_ranging & (regimes_temp == "Indeterminate")] = "Ranging_Normal"  # Default for range if no strong vol signal
        regimes_temp[is_transitioning_adx & (regimes_temp == "Indeterminate")] = "Transition_Normal"


        regimes = regimes_temp # Assign the result


        # logger.info(f"Market regimes classified. Counts: {regimes.value_counts().to_dict()}") # Too verbose
        return regimes


# Note: classify_market_regime_trinary is defined above in Section II as part of the Trinary System functions.

# ===== WRAPPER FOR CONSISTENT API (already present) =====
# def initialize(config: ConfigParser): ... (already present)
# This function is already defined above and calls initialize_indicators.
