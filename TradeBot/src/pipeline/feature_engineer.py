# feature_engineer.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List, Optional, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import os
from configparser import ConfigParser
from src import indicators # Import the module, but we need an instance for methods

logger = logging.getLogger(__name__)

# --- Gestion des Scalers (fonctions de module) ---
_scalers: Dict[str, MinMaxScaler | StandardScaler] = {}
_scalers_file: Optional[str] = None

def initialize_feature_scalers(config: ConfigParser):
    """Initialise le chemin du fichier des scalers et charge les scalers si existants."""
    global _scalers_file
    _scalers_file = config.get('training', 'scalers_file', fallback='scalers.pkl')
    load_scalers()
    logger.info(f"Feature scalers initialized. File: {_scalers_file}")

def load_scalers():
    """Charge les scalers depuis le fichier."""
    global _scalers
    if _scalers_file and os.path.exists(_scalers_file):
        try:
            with open(_scalers_file, 'rb') as f:
                _scalers = pickle.load(f)
                logger.info(f"Scalers chargés depuis {_scalers_file}. Scalers chargés pour: {list(_scalers.keys())}")
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Erreur lors du chargement des scalers depuis {_scalers_file}: {e}")
            _scalers = {}
        except Exception as e:
            logger.error(f"Erreur inattendue lors du chargement des scalers: {e}", exc_info=True)
            _scalers = {}
    else:
        _scalers = {}
        logger.info("Aucun fichier de scalers trouvé. Ils seront entraînés lors du premier cycle.")

def save_scalers():
    """Sauvegarde les scalers entraînés."""
    if _scalers_file and _scalers:
        try:
            dirname = os.path.dirname(_scalers_file)
            if dirname and not os.path.exists(dirname):
                 os.makedirs(dirname)
            with open(_scalers_file, 'wb') as f:
                pickle.dump(_scalers, f)
            logger.debug(f"Scalers sauvegardés dans {_scalers_file}.")
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Erreur lors de la sauvegarde des scalers dans {_scalers_file}: {e}")
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la sauvegarde des scalers: {e}", exc_info=True)

def get_scaler(name: str):
    """Retourne un scaler par nom."""
    return _scalers.get(name)

def fit_and_save_scalers(data: pd.DataFrame):
    """Entraîne les scalers sur les données fournies et les sauvegarde."""
    global _scalers
    if data.empty:
        logger.warning("Données vides fournies pour entraîner les scalers.")
        return

    logger.info(f"Entraînement des scalers sur {len(data)} lignes et {len(data.columns)} colonnes...")
    _scalers = {}
    
    feature_cols = data.columns.tolist()

    scaler_features = StandardScaler()
    _scalers['feature_scaler'] = scaler_features.fit(data[feature_cols])
    logger.info(f"Scaler principal entraîné pour {len(feature_cols)} features.")

    save_scalers()


class FeatureEngineer:
    def __init__(self, config_manager: ConfigParser, indicators_manager_instance: indicators.IndicatorsManager):
        self.config = config_manager
        self.indicators_manager = indicators_manager_instance # Store the instance
        initialize_feature_scalers(self.config)
        logger.info("FeatureEngineer initialized.")

    def get_scaler(self, name: str = 'feature_scaler') -> Optional[Any]:
        """Returns the main feature scaler."""
        return get_scaler(name)

    def build_complete_raw_features(
        self,
        ohlcv_row_series: pd.Series,
        ms_features: Dict,    
        of_features: Dict,    
        pi_ratings: Dict,
        ta_features_row: pd.Series,
        indicator_config: Dict
    ) -> pd.Series:
        """
        Combine toutes les features brutes/calculées pour un pas de temps unique en une seule Series.
        Cette Series représente le vecteur de features AVANT la création de features dérivées ou de séquence.
        C'est la donnée de base passée au scaler.
        """
        features = ohlcv_row_series.copy()

        if ta_features_row is not None and not ta_features_row.empty:
            for ta_key, ta_value in ta_features_row.items():
                if ta_key not in features:
                    features[ta_key] = ta_value
                elif pd.isna(features[ta_key]) and not pd.isna(ta_value):
                     features[ta_key] = ta_value

        for key, value in ms_features.items():
            if isinstance(value, list):
                if value:
                    features[f"{key}_count"] = len(value)
                    features[f"{key}_min"] = min(value) if value else np.nan
                    features[f"{key}_max"] = max(value) if value else np.nan
                    features[f"{key}_mean"] = np.mean(value) if value else np.nan
                    if 'close' in features:
                        current_price = features['close']
                        closest_price = min(value, key=lambda x: abs(x - current_price)) if value else np.nan
                        features[f"{key}_dist_to_closest"] = current_price - closest_price if closest_price is not np.nan else np.nan
                else:
                    features[f"{key}_count"] = 0
                    features[f"{key}_min"] = np.nan
                    features[f"{key}_max"] = np.nan
                    features[f"{key}_mean"] = np.nan
                    features[f"{key}_dist_to_closest"] = np.nan
            else:
                features[key] = value

        for key, value in of_features.items():
            features[key] = value

        rh_col = pi_ratings.get('R_H', 0.0) # Use pi_ratings directly
        ra_col = pi_ratings.get('R_A', 0.0) # Use pi_ratings directly
        features['R_Diff'] = rh_col - ra_col
        features['R_Ratio'] = rh_col / (ra_col + 1e-9) 
        
        confidence_score = self.indicators_manager.calculate_confidence_score(rh_col, ra_col) # Use instance method
        features['confidence_score'] = confidence_score

        rsi_value = ta_features_row.get('RSI', np.nan) 
        if pd.isna(rsi_value):
            rsi_period_config = indicator_config.get('rsi_period', self.indicators_manager.DEFAULT_TRINARY_RSI_PERIOD) # Use instance default
            rsi_value = ta_features_row.get(f'RSI_{rsi_period_config}', 50.0)
        
        market_momentum = (rsi_value - 50.0) / 50.0
        market_momentum = np.clip(market_momentum, -1.0, 1.0)
        features['market_momentum'] = market_momentum

        atr_stability_period_config = indicator_config.get('trinary_atr_period_stability', self.indicators_manager.DEFAULT_TRINARY_ATR_PERIOD_STABILITY) # Use instance default
        atr_stability_value = ta_features_row.get(f'ATR_{atr_stability_period_config}', np.nan)
        if pd.isna(atr_stability_value):
            atr_stability_value = ta_features_row.get('ATR', np.nan)

        close_price = ohlcv_row_series.get('close', np.nan)
        
        stability_factor = np.nan
        if not pd.isna(atr_stability_value) and not pd.isna(close_price) and close_price != 0:
            normalized_atr = close_price / (atr_stability_value + 1e-9)
            stability_factor = normalized_atr
        else:
            stability_factor = 0.5
        
        features['stability_factor_ift_input'] = stability_factor

        r_h_for_ift = pi_ratings.get('R_H', features.get('R_H', 0.0))
        r_a_for_ift = pi_ratings.get('R_A', features.get('R_A', 0.0))
        
        r_ift_trend, r_ift_strength = self.indicators_manager.calculate_ift(r_h_for_ift, r_a_for_ift, market_momentum, stability_factor) # Use instance method
        features['R_IFT_Trend'] = r_ift_trend
        features['R_IFT_Strength'] = r_ift_strength

        sentiment_score = ohlcv_row_series.get('sentiment_score', 0.0)
        base_stability_index = np.clip(abs(sentiment_score), 0.0, 1.0)
        features['base_stability_index_rdvu_input'] = base_stability_index

        atr_vol_period_config = indicator_config.get('trinary_atr_period_volatility', self.indicators_manager.DEFAULT_TRINARY_ATR_PERIOD_VOLATILITY) # Use instance default
        atr_sma_period_config = indicator_config.get('trinary_atr_sma_period', self.indicators_manager.DEFAULT_TRINARY_ATR_SMA_PERIOD) # Use instance default
        
        atr_current_col_name = f'ATR_TRINARY_VOLATILITY_{atr_vol_period_config}'
        features['atr_current_rdvu_input'] = ta_features_row.get(atr_current_col_name, np.nan)

        atr_average_col_name = f'ATR_SMA_TRINARY_VOLATILITY_{atr_vol_period_config}_{atr_sma_period_config}'
        features['atr_average_rdvu_input'] = ta_features_row.get(atr_average_col_name, np.nan)

        if 'ms_va_high_price' in features and features['ms_va_high_price'] is not None and 'of_cvd' in features:
            if 'close' in features and features['close'] > 0:
                close_price = features['close']
                va_high = features['ms_va_high_price']
                dist_pct_to_va_high = (va_high - close_price) / close_price
                if 0 < dist_pct_to_va_high < 0.005 and features['of_cvd'] > 0:
                    features['ms_of_va_high_breakout_potential'] = features['of_cvd'] * (1 - dist_pct_to_va_high * 200)
                else:
                    features['ms_of_va_high_breakout_potential'] = 0.0
            else:
                features['ms_of_va_high_breakout_potential'] = 0.0
        
        if 'ms_va_low_price' in features and features['ms_va_low_price'] is not None and 'of_cvd' in features:
            if 'close' in features and features['close'] > 0:
                close_price = features['close']
                va_low = features['ms_va_low_price']
                dist_pct_to_va_low = (close_price - va_low) / close_price
                if 0 < dist_pct_to_va_low < 0.005 and features['of_cvd'] < 0:
                    features['ms_of_va_low_breakdown_potential'] = -features['of_cvd'] * (1 - dist_pct_to_va_low * 200)
                else:
                    features['ms_of_va_low_breakdown_potential'] = 0.0
            else:
                features['ms_of_va_low_breakdown_potential'] = 0.0
        else:
            features['ms_of_va_low_breakdown_potential'] = 0.0

        return features


    def build_feature_vector_for_xgboost(
        self,
        complete_raw_features: pd.Series,
        xgb_features_names: List[str],
        scaler: Optional[MinMaxScaler | StandardScaler] = None
    ) -> Optional[pd.DataFrame]:
        """
        Prépare le vecteur de features final, normalisé pour la prédiction XGBoost.
        Sélectionne les colonnes de complete_raw_features correspondant à xgb_features_names, aligne l'ordre, normalise,
        et retourne un DataFrame [1, num_xgb_features].
        """
        if complete_raw_features is None or xgb_features_names is None:
            logger.error("Entrées manquantes pour la construction du vecteur de features XGBoost.")
            return None

        try:
            features_for_xgb = pd.Series(index=xgb_features_names)
            
            for col in xgb_features_names:
                if col in complete_raw_features:
                    features_for_xgb[col] = complete_raw_features[col]
                else:
                    logger.warning(f"Colonne de feature '{col}' manquante pour XGBoost. Utilisation de 0 comme fallback.")
                    features_for_xgb[col] = 0.0
            
            features_t_xgb_df = pd.DataFrame([features_for_xgb], index=[complete_raw_features.name])
            
            if scaler:
                features_t_xgb_scaled_np = scaler.transform(features_t_xgb_df)
                features_t_xgb_scaled_df = pd.DataFrame(features_t_xgb_scaled_np, index=features_t_xgb_df.index, columns=xgb_features_names)
                
                if features_t_xgb_scaled_df.isnull().values.any():
                    logger.error("NaNs détectés dans les features XGBoost normalisées.")
                    features_t_xgb_scaled_df.fillna(0, inplace=True)
                
                return features_t_xgb_scaled_df
            else:
                logger.warning("Aucun scaler fourni. Retourne des features XGBoost non normalisées.")
                return features_t_xgb_df

        except KeyError as e:
            logger.error(f"Colonne de feature manquante ou incorrecte pour XGBoost: {e}. Vérifiez 'training_params.json' et les colonnes calculées.")
            return None
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la construction du vecteur de features XGBoost: {e}", exc_info=True)
            return None


    def build_feature_sequence_for_fq(
        self,
        df_historical_features: pd.DataFrame,
        fq_config: Dict,
        scaler: Optional[MinMaxScaler | StandardScaler],
        all_feature_columns: List[str],
        current_ts: pd.Timestamp
    ) -> Optional[np.ndarray]:
        """
        Construit une séquence de features pour le modèle FutureQuant.
        Prend le DataFrame de données historiques avec toutes les features.
        Retourne un array numpy de forme [1, window_size_in, num_features] prêt pour la prédiction.
        """
        if df_historical_features is None or df_historical_features.empty or scaler is None or not all_feature_columns:
            logger.error("Entrées manquantes ou vides pour construction de séquence FQ.")
            return None

        try:
            fq_window_size_in = int(fq_config.get('window_size_in', 30))
        except ValueError:
            logger.error("Valeur invalide pour 'window_size_in'.")
            return None

        try:
            current_idx_loc = df_historical_features.index.get_loc(current_ts)
        except KeyError:
            logger.error(f"Timestamp {current_ts} non trouvé dans l'index des données historiques.")
            return None

        seq_start_idx_loc = current_idx_loc - fq_window_size_in + 1
        seq_end_idx_loc = current_idx_loc

        if seq_start_idx_loc < 0:
            logger.warning(f"Pas assez de données ({current_idx_loc + 1} < {fq_window_size_in}) pour former une séquence FQ pour {current_ts}.")
            return None

        try:
            sequence_data = df_historical_features.iloc[seq_start_idx_loc:seq_end_idx_loc + 1].copy()
            
            for col in all_feature_columns:
                if col not in sequence_data.columns:
                    sequence_data[col] = 0.0
                    logger.warning(f"Colonne de feature '{col}' manquante dans les données de séquence. Ajoutée avec des zéros.")
            
            sequence_raw_df = sequence_data[all_feature_columns].copy()

            if sequence_raw_df.isnull().values.any():
                logger.warning(f"NaNs détectés dans la séquence FQ pour {current_ts}. Remplissage avec 0.")
                sequence_raw_df.fillna(0, inplace=True)

            sequence_scaled_np = scaler.transform(sequence_raw_df.values)
            X_live_fq_seq = np.expand_dims(sequence_scaled_np, axis=0)
            logger.debug(f"Séquence FQ construite pour {current_ts} avec forme {X_live_fq_seq.shape}.")

            return X_live_fq_seq

        except KeyError as e:
            logger.error(f"Colonne de feature manquante pour construction de séquence FQ: {e}. Vérifiez 'training_params.json'.")
            return None
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la construction de la séquence FQ: {e}", exc_info=True)
            return None