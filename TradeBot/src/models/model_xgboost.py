# model_xgboost.py
import xgboost as xgb
import logging
import pandas as pd
import numpy as np # Added numpy import
import json
import os
from typing import Optional, List, Dict, Tuple
from configparser import ConfigParser, NoSectionError, NoOptionError # Added imports

logger = logging.getLogger(__name__)

# --- Gestion du Modèle XGBoost ---
_xgb_model: Optional[xgb.Booster] = None
_xgb_model_file: Optional[str] = None
_xgb_trained_columns: Optional[List[str]] = None
_xgb_columns_file: Optional[str] = None


def initialize(config: ConfigParser):
    """Wrapper function for initialize_xgboost_model to maintain compatibility with trading_analyzer.py."""
    initialize_xgboost_model(config)


def initialize_xgboost_model(config: ConfigParser):
    """Initialise les chemins des fichiers et charge le modèle/colonnes si existants."""
    global _xgb_model_file, _xgb_columns_file
    _xgb_model_file = config.get('xgboost', 'model_file', fallback='xgb_model.json')
    _xgb_columns_file = config.get('training', 'training_params_file', fallback='training_params.json') # Colonnes stockées dans le fichier de params global

    load_xgboost_model()
    logger.info(f"XGBoost model initialized. Model file: {_xgb_model_file}, Columns file: {_xgb_columns_file}")

def load_xgboost_model():
    """Charge le modèle XGBoost et les noms de colonnes depuis les fichiers."""
    global _xgb_model, _xgb_trained_columns
    _xgb_model = None
    _xgb_trained_columns = None

    # Charger les colonnes
    if _xgb_columns_file and os.path.exists(_xgb_columns_file):
        try:
            with open(_xgb_columns_file, 'r') as f:
                params = json.load(f)
                _xgb_trained_columns = params.get('xgboost_features')
                if isinstance(_xgb_trained_columns, list):
                    logger.info(f"XGBoost trained columns ({len(_xgb_trained_columns)}) loaded from {_xgb_columns_file}.")
                else:
                    logger.warning(f"Clé 'xgboost_features' manquante ou invalide dans {_xgb_columns_file}.")
                    _xgb_trained_columns = None
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Erreur lors du chargement des colonnes XGBoost depuis {_xgb_columns_file}: {e}")
            _xgb_trained_columns = None
        except Exception as e:
            logger.error(f"Erreur inattendue lors du chargement des colonnes XGBoost: {e}", exc_info=True)
            _xgb_trained_columns = None


    # Charger le modèle
    if _xgb_model_file and os.path.exists(_xgb_model_file):
        try:
            model = xgb.Booster()
            model.load_model(_xgb_model_file)
            _xgb_model = model
            logger.info(f"XGBoost model chargé avec succès depuis: {_xgb_model_file}")
        except xgb.core.XGBoostError as e:
            logger.warning(f"Impossible de charger le modèle XGBoost depuis '{_xgb_model_file}'. Le fichier n'existe peut-être pas ou est invalide: {e}")
            _xgb_model = None
        except Exception as e:
            logger.error(f"Erreur inattendue lors du chargement du modèle XGBoost: {e}", exc_info=True)
            _xgb_model = None

    # Vérifier la cohérence
    if _xgb_model and not _xgb_trained_columns:
         logger.warning("Modèle XGBoost chargé mais colonnes d'entraînement manquantes. Le modèle ne pourra pas être utilisé pour la prédiction.")
         _xgb_model = None # Invalider le modèle si les colonnes sont inconnues
    elif not _xgb_model and _xgb_trained_columns:
         logger.warning("Colonnes d'entraînement XGBoost chargées mais modèle manquant.")


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, config: ConfigParser) -> Optional[xgb.Booster]:
    """Entraîne un modèle XGBoost classifieur."""
    global _xgb_trained_columns # On mettra à jour les colonnes utilisées pour l'entraînement

    if X_train.empty or y_train.empty or len(X_train) != len(y_train):
        logger.error("Données d'entraînement (X ou y) vides ou de tailles incohérentes. Entraînement XGBoost annulé.")
        return None

    try:
        # Construire les paramètres depuis la config
        params = {
            'objective': config.get('xgboost', 'objective', fallback='multi:softmax'),
            'num_class': config.getint('xgboost', 'num_class', fallback=3),
            'max_depth': config.getint('xgboost', 'max_depth', fallback=5),
            'learning_rate': config.getfloat('xgboost', 'learning_rate', fallback=0.1),
            'eval_metric': config.get('xgboost', 'eval_metric', fallback='mlogloss'),
            'tree_method': 'hist', # Souvent plus rapide
            'subsample': config.getfloat('xgboost', 'subsample', fallback=0.8),
            'colsample_bytree': config.getfloat('xgboost', 'colsample_bytree', fallback=0.8),
            'seed': 42 # Pour la reproductibilité
            # Ajouter d'autres paramètres XGBoost ici si nécessaire
        }
        n_estimators = config.getint('xgboost', 'n_estimators', fallback=150)

        logger.info(f"Début de l'entraînement XGBoost avec {len(X_train)} échantillons, {len(X_train.columns)} features.")
        logger.debug(f"Paramètres XGBoost: {params}")
        logger.debug(f"Nombre d'estimateurs: {n_estimators}")

        # Créer la DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # Entraînement
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            # Ajouter watchlist et early stopping si désiré (nécessite X_val, y_val)
            # evals=[(dtrain, 'train')], # Optionnel pour debug
            # verbose_eval=50 # Optionnel pour suivre
        )
        logger.info("Entraînement du modèle XGBoost terminé avec succès.")

        # Mettre à jour les colonnes utilisées pour l'entraînement
        _xgb_trained_columns = X_train.columns.tolist()
        # La sauvegarde des colonnes se fera dans training_pipeline

        return model

    except (ValueError, TypeError, NoSectionError, NoOptionError) as e:
        logger.error(f"Erreur de configuration XGBoost: {e}.")
        return None
    except xgb.core.XGBoostError as e:
        logger.error(f"Erreur interne XGBoost lors de l'entraînement: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'entraînement XGBoost: {e}", exc_info=True)
        return None

def save_xgboost_model(model: xgb.Booster):
    """Sauvegarde le modèle XGBoost entraîné."""
    if model and _xgb_model_file:
        try:
            dirname = os.path.dirname(_xgb_model_file)
            if dirname and not os.path.exists(dirname):
                 os.makedirs(dirname)
            model.save_model(_xgb_model_file)
            logger.info(f"Modèle XGBoost sauvegardé sous: {_xgb_model_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle XGBoost dans {_xgb_model_file}: {e}", exc_info=True)
    elif not model:
         logger.warning("Tentative de sauvegarde d'un modèle XGBoost None.")
    elif not _xgb_model_file:
         logger.error("Nom de fichier non fourni pour la sauvegarde du modèle XGBoost.")


def predict_xgboost(X_live: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Fait une prédiction de probabilité (Buy/Sell/Hold) en utilisant le modèle XGBoost chargé.
    Retourne un array numpy [prob_hold, prob_buy, prob_sell] ou None.
    """
    if _xgb_model is None or _xgb_trained_columns is None:
        logger.error("Modèle XGBoost ou colonnes d'entraînement non chargés.")
        return None
    if X_live.empty:
        logger.warning("DataFrame d'entrée vide pour la prédiction XGBoost.")
        return None

    # --- Alignement des colonnes ---
    # S'assurer que les colonnes de X_live correspondent à celles utilisées à l'entraînement
    current_cols = set(X_live.columns)
    expected_cols = set(_xgb_trained_columns)

    missing_cols = list(expected_cols - current_cols)
    extra_cols = list(current_cols - expected_cols)

    if missing_cols:
        logger.warning(f"Colonnes manquantes pour prédiction XGBoost: {missing_cols}. Remplissage avec 0.0.")
        for col in missing_cols:
            X_live[col] = 0.0 # Remplir les colonnes manquantes avec 0 (ou autre valeur par défaut)

    if extra_cols:
        logger.warning(f"Colonnes supplémentaires ignorées pour prédiction XGBoost: {extra_cols}.")
        X_live = X_live.drop(columns=extra_cols)

    # Réorganiser les colonnes dans le même ordre que l'entraînement
    try:
        X_live_aligned = X_live[_xgb_trained_columns]
    except KeyError as e:
        logger.error(f"Erreur critique lors de la réorganisation des colonnes pour prédiction XGBoost: Colonne '{e}' attendue mais non trouvée après remplissage.")
        return None

    # Vérifier les NaNs restants après alignement et remplissage
    if X_live_aligned.isnull().values.any():
         logger.error("NaNs détectés dans les features finales alignées pour prédiction XGBoost.")
         # Option: tenter de remplir avec 0 ou moyenne si c'est tolérable pour ce modèle
         # X_live_aligned = X_live_aligned.fillna(0.0) # Risqué
         return None # Préférable d'échouer si données invalides

    logger.debug(f"Prédiction XGBoost pour 1 échantillon avec {len(X_live_aligned.columns)} features.")

    try:
        dmatrix_live = xgb.DMatrix(X_live_aligned)
        # Pour multi:softmax, predict retourne la classe directement.
        # Pour obtenir les probabilités, il faut utiliser predict_proba si c'est un wrapper sklearn,
        # ou spécifier output_margin=True et passer dans softmax manuellement avec l'API native xgb.Booster.
        # L'objectif 'multi:softmax' est plus simple pour avoir directement la classe.
        # Si on veut les probabilités, l'objectif 'multi:softprob' est mieux.
        # Supposons qu'on utilise 'multi:softprob' comme objectif dans la config pour obtenir les probas.
        # Changeons l'objectif par défaut dans la config et ici.

        # Si objective est 'multi:softprob', predict retourne un array de shape [n_samples, n_classes] avec les probas.
        params_obj = _xgb_model.get_param('objective')
        if params_obj == 'multi:softprob':
             prediction_probs = _xgb_model.predict(dmatrix_live) # Shape [1, 3]
             logger.debug(f"Prédiction probas XGBoost ({params_obj}): {prediction_probs[0].tolist()}")
             return prediction_probs[0] # Retourne array [p_hold, p_buy, p_sell]
        elif params_obj == 'multi:softmax':
            # Si l'objectif est multi:softmax, on peut soit retourner la classe prédite (int), soit
            # simuler les probas en mettant 1 pour la classe prédite et 0 pour les autres.
            # Pour la stratégie hybride, les probas sont plus utiles.
            # Assumons multi:softprob est utilisé dans la config pour le besoin de probas.
             predicted_class = int(_xgb_model.predict(dmatrix_live)[0])
             logger.warning(f"XGBoost objective est '{params_obj}'. La stratégie hybride attend multi:softprob pour les probas. Retourne probas simplifiées.")
             probs = np.zeros(3)
             probs[predicted_class] = 1.0
             return probs # Retourne array [p_hold, p_buy, p_sell]

        else:
            logger.error(f"Objective XGBoost non supporté pour la prédiction de probabilités : {params_obj}")
            return None


    except xgb.core.XGBoostError as e:
        logger.error(f"Erreur interne XGBoost lors de la prédiction: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la prédiction XGBoost: {e}", exc_info=True)
        return None 

def get_model():
    """
    Retourne le modèle XGBoost chargé
    """
    return _xgb_model 