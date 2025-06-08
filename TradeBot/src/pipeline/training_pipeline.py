# training_pipeline.py
import pandas as pd
import numpy as np
import logging
import os
import json
import time
import pickle
import traceback
from typing import Dict, List, Tuple, Optional
from configparser import ConfigParser, NoSectionError, NoOptionError
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import data_handler
from analysis import sentiment_analyzer
import indicators
from pipeline import feature_engineer
from pipeline import labelling
from models import model_xgboost
from models import model_futurequant
from core import db_handler
from analysis import market_structure_analyzer
from analysis import order_flow_analyzer

logger = logging.getLogger(__name__)

def load_data_from_database(config: ConfigParser, symbol_market: str, start_date, end_date) -> Optional[pd.DataFrame]:
    """
    Charge les données historiques depuis la base de données pour un actif spécifique.
    
    Args:
        config: Configuration du système
        symbol_market: Identifiant de l'actif (ex: 'BTC/USD@ccxt')
        start_date: Date de début pour les données historiques
        end_date: Date de fin pour les données historiques
        
    Returns:
        DataFrame pandas avec les données OHLCV ou None en cas d'erreur
    """
    try:
        # Obtenir l'engine de la base de données
        engine = db_handler.get_db_engine(config)
        if engine is None:
            logger.error("Impossible d'obtenir l'engine de la base de données.")
            return None
        
        # Extraire symbol et market de symbol_market
        if '@' in symbol_market:
            symbol, market = symbol_market.split('@')
        else:
            symbol = symbol_market
            market = config.get('trading', 'default_market', fallback='ccxt') # Utiliser un fallback depuis config
        
        # Récupérer les données OHLCV
        logger.info(f"Récupération des données pour {symbol} sur {market} entre {start_date} et {end_date}")
        df_ohlcv = db_handler.get_historical_ohlcv(engine, symbol, market, start_date, end_date)
        
        if df_ohlcv is None or df_ohlcv.empty:
            logger.warning(f"Aucune donnée OHLCV trouvée pour {symbol_market}")
            return None
            
        logger.info(f"Données récupérées: {len(df_ohlcv)} barres")

        # Obtenir le nom de la colonne de sentiment configuré
        sentiment_col_name = config.get('backtest', 'sentiment_col', fallback='sentiment_score')
        
        # Récupérer les données sentiment via sentiment_analyzer
        df_sentiment = sentiment_analyzer.get_sentiment_data(symbol, market, start_date, end_date)
        
        if df_sentiment is not None and not df_sentiment.empty:
            # S'assurer que l'index de df_sentiment est DatetimeIndex et UTC pour la fusion correcte
            if not isinstance(df_sentiment.index, pd.DatetimeIndex):
                df_sentiment.index = pd.to_datetime(df_sentiment.index, utc=True)
            elif df_sentiment.index.tzinfo is None:
                df_sentiment.index = df_sentiment.index.tz_localize('UTC')
            else:
                df_sentiment.index = df_sentiment.index.tz_convert('UTC')

            # S'assurer que l'index de df_ohlcv est aussi DatetimeIndex et UTC
            if not isinstance(df_ohlcv.index, pd.DatetimeIndex):
                df_ohlcv.index = pd.to_datetime(df_ohlcv.index, utc=True)
            elif df_ohlcv.index.tzinfo is None:
                df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
            else:
                df_ohlcv.index = df_ohlcv.index.tz_convert('UTC')

            df_ohlcv = pd.merge_asof(df_ohlcv.sort_index(), df_sentiment.sort_index(), 
                                     left_index=True, right_index=True, 
                                     direction='nearest', tolerance=pd.Timedelta('1D')) # Tolérance pour fusion
            
            # Renommer la colonne 'sentiment' si elle existe et si elle est différente de sentiment_col_name
            if 'sentiment' in df_ohlcv.columns and sentiment_col_name != 'sentiment':
                df_ohlcv.rename(columns={'sentiment': sentiment_col_name}, inplace=True)
            elif 'sentiment' not in df_ohlcv.columns and sentiment_col_name not in df_ohlcv.columns:
                # Si ni 'sentiment' ni sentiment_col_name n'est présent après la fusion (cas improbable si df_sentiment avait 'sentiment')
                logger.warning(f"Colonne de sentiment ('sentiment' ou '{sentiment_col_name}') non trouvée après fusion pour {symbol_market}. Ajout de la colonne par défaut '{sentiment_col_name}'.")
                df_ohlcv[sentiment_col_name] = 0.0
            # S'assurer que la colonne finale a le bon nom et remplir les NaNs introduits par merge_asof
            if sentiment_col_name in df_ohlcv.columns:
                df_ohlcv[sentiment_col_name].fillna(0.0, inplace=True)
            else: # Si la colonne n'existe toujours pas (ex: df_sentiment était vide, merge n'a rien fait)
                df_ohlcv[sentiment_col_name] = 0.0
        else:
            logger.warning(f"Pas de données sentiment pour {symbol_market}. Ajout de la colonne par défaut '{sentiment_col_name}'.")
            df_ohlcv[sentiment_col_name] = 0.0
        
        # Récupérer les données order flow
        df_order_flow = db_handler.get_order_flow_data(symbol, market, start_date, end_date)
        if df_order_flow is not None and not df_order_flow.empty:
            df_ohlcv['of_data_available'] = True
            # On ne fusionne pas les données OF directement, car elles seront traitées séparément
            # par order_flow_analyzer.py
        else:
            logger.warning(f"Pas de données order flow pour {symbol_market}")
            df_ohlcv['of_data_available'] = False
        
        return df_ohlcv
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données depuis la base: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def prepare_historical_data(config: ConfigParser, apis: Dict) -> Dict[str, pd.DataFrame]:
    """
    Prépare les données historiques complètes avec toutes les features pour tous les actifs.
    
    Args:
        config: Configuration du système
        apis: Dictionnaire d'APIs externes (non utilisé ici, pour compatibilité)
        
    Returns:
        Dictionnaire {symbol_market: DataFrame} avec les données préparées
    """
    # Initialiser le module d'analyse de sentiment
    sentiment_analyzer.initialize_sentiment_analyzer(config)

    # Récupérer les paramètres de configuration
    assets_str = config.get('trading', 'assets', fallback='')  
    assets = [asset.strip() for asset in assets_str.split(',') if asset.strip()]
    historical_data_years = config.getint('training', 'historical_data_years', fallback=1) 
    lookback_days = historical_data_years * 365  
    
    if not assets:
        logger.error("Aucun actif configuré pour l'entraînement.")
        return {}
    
    # Calculer les dates de début et fin
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    all_prepared_data = {}
    
    # Initialiser les modules avec la configuration
    indicators.initialize(config)
    feature_engineer.initialize_feature_scalers(config)
    order_flow_analyzer.initialize(config)
    market_structure_analyzer.initialize(config)
    
    # Traiter chaque actif
    for symbol_market in assets:
        try:
            logger.info(f"Préparation des données pour {symbol_market}")
            
            # 1. Charger les données brutes OHLCV + sentiment
            df_historical = load_data_from_database(config, symbol_market, start_date, end_date)
            
            if df_historical is None or df_historical.empty:
                logger.warning(f"Données manquantes pour {symbol_market}, actif ignoré.")
                continue

            # 2. Générer ou mettre à jour les Pi-Ratings
            for i in range(1, len(df_historical)):
                prev_price = df_historical['close'].iloc[i-1]
                curr_price = df_historical['close'].iloc[i]
                timestamp = df_historical.index[i]
                
                # Mise à jour des ratings en mode historique (pas de sauvegarde)
                indicators.update_ratings(symbol_market, curr_price, prev_price, timestamp)
            
            # Récupérer les ratings finaux après mise à jour complète
            ratings = indicators.get_ratings(symbol_market)
            ratings_history = indicators.get_ratings_history(symbol_market)
            
            # Convertir l'historique des ratings en DataFrame et fusionner avec les données historiques
            if ratings_history:
                df_ratings = pd.DataFrame(ratings_history)
                df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'])
                df_ratings.set_index('timestamp', inplace=True)
                
                # Aligner les données en utilisant l'index
                df_historical = df_historical.join(df_ratings[['R_H', 'R_A', 'confidence']], how='left')
                
                # Remplir les valeurs manquantes (périodes sans mise à jour)
                df_historical['R_H'].fillna(method='ffill', inplace=True)
                df_historical['R_A'].fillna(method='ffill', inplace=True)
                df_historical['confidence'].fillna(method='ffill', inplace=True)
                
                # Remplir les NaN au début avec 0.0
                df_historical['R_H'].fillna(0.0, inplace=True)
                df_historical['R_A'].fillna(0.0, inplace=True)
                df_historical['confidence'].fillna(0.0, inplace=True)
            else:
                logger.warning(f"Pas d'historique de ratings pour {symbol_market}")
                df_historical['R_H'] = 0.0
                df_historical['R_A'] = 0.0
                df_historical['confidence'] = 0.0
            
            # 3. Calculer les features TA classiques
            indicator_config = dict(config['indicators']) if 'indicators' in config.sections() else {}
            df_ta = indicators.calculate_all_ta_indicators(df_historical, indicator_config)
            
            # Fusionner les features TA avec le DataFrame historique
            df_historical = pd.concat([df_historical, df_ta], axis=1)
            
            # 4. Charger et analyser les données Order Flow (si disponibles)
            of_data_dict = {}
            has_order_flow = df_historical.get('of_data_available', False).any()
            
            if has_order_flow:
                # Récupérer les données OF depuis la base de données
                for idx, row in df_historical.iterrows():
                    # Pour chaque barre, nous récupérons les données OF correspondantes
                    # (dans une implémentation réelle, cela pourrait être optimisé en chargeant par blocs)
                    start_time = idx - timedelta(hours=1)  # Adapter à la taille de la barre
                    end_time = idx
                    
                    # Récupérer les données OF pour cette période précise
                    of_data_period = db_handler.get_order_flow_tick_data(symbol_market, start_time, end_time)
                    if of_data_period is not None and not of_data_period.empty:
                        of_data_dict[idx] = of_data_period
            
            # 5. Calculer les features MS et OF pour chaque barre
            all_ms_features = {}
            all_of_features = {}
            
            for idx, row in df_historical.iterrows():
                # Calculer les features de Market Structure pour chaque timestamp
                historical_window = df_historical.loc[:idx]  # Toutes les données jusqu'à l'index actuel
                ms_features = market_structure_analyzer.calculate_market_structure_features(
                    historical_window, config
                )
                all_ms_features[idx] = ms_features
                
                # Calculer les features Order Flow pour chaque timestamp (si disponibles)
                if idx in of_data_dict:
                    of_features = order_flow_analyzer.analyze_order_flow_for_period(
                        of_data_dict[idx], config
                    )
                else:
                    # Features par défaut si pas de données OF
                    of_features = {
                        'of_cvd': 0.0,
                        'of_absorption_score': 0.0,
                        'of_trapped_traders_bias': 0.0
                    }
                
                all_of_features[idx] = of_features
            
            # 6. Construire le DataFrame complet de features
            complete_features_df = df_historical.copy()
            
            # Ajouter les features MS et OF
            for idx in complete_features_df.index:
                ms_features = all_ms_features.get(idx, {})
                of_features = all_of_features.get(idx, {})
                
                # Construire les raw features avec MS et OF
                complete_raw_features = feature_engineer.build_complete_raw_features(
                    complete_features_df.loc[idx], ms_features, of_features
                )
                
                # Mettre à jour le DataFrame avec les features calculées
                for col, value in complete_raw_features.items():
                    if col not in complete_features_df.columns:
                        complete_features_df.loc[idx, col] = value
            
            # 7. Ajouter les labels et targets pour l'entraînement
            # Configuration du labelling
            label_window = config.getint('training', 'label_window_periods', fallback=10)
            label_threshold = config.getfloat('training', 'label_price_threshold', fallback=0.01)
            
            # Calculer les labels discrets (pour XGBoost) et ratios de prix futurs (pour FutureQuant)
            complete_features_df = labelling.add_labels_to_dataframe(
                complete_features_df, label_window, label_threshold
            )
            
            # Supprimer les NaNs (générés par le labelling sur les dernières barres)
            complete_features_df.dropna(subset=['discrete_label', 'future_price_ratio_target'], inplace=True)
            
            # Stocker les données préparées pour l'actif
            all_prepared_data[symbol_market] = complete_features_df
            logger.info(f"Données préparées pour {symbol_market}: {len(complete_features_df)} barres avec {len(complete_features_df.columns)} features")
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des données pour {symbol_market}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Sauvegarder les données préparées pour une utilisation future (analyse, backtesting)
    if all_prepared_data:
        data_dir = config.get('paths', 'data_dir', fallback='data')
        os.makedirs(data_dir, exist_ok=True)
        prepared_data_file = config.get('training', 'prepared_data_file', fallback='prepared_data.pickle')
        prepared_data_path = os.path.join(data_dir, prepared_data_file)
        
        try:
            with open(prepared_data_path, 'wb') as f:
                pickle.dump(all_prepared_data, f)
            logger.info(f"Données préparées sauvegardées dans {prepared_data_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données préparées: {e}")
    
    return all_prepared_data

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
            logger.info(f"Paramètres d'entraînement chargés depuis {training_params_file}")
        else:
            logger.warning(f"Fichier de paramètres d'entraînement {training_params_file} non trouvé")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des paramètres d'entraînement: {e}")
    
    # Charger le scaler
    scaler = feature_engineer.load_feature_scaler()
    
    # Charger les modèles
    xgb_model = model_xgboost.get_model()
    fq_model = model_futurequant.get_model()
    
    return {
        'scaler': scaler,
        'xgb': xgb_model,
        'fq': fq_model,
        'params': loaded_training_params
    }

def run_training_pipeline(config: ConfigParser, apis: Dict):
    """
    Exécute le pipeline complet: prépare les données historiques, entraîne les scalers,
    puis entraîne les modèles XGBoost et FutureQuant. Sauvegarde les modèles et scalers.
    """
    logger.info("="*40)
    logger.info(" DÉBUT DU PIPELINE D'ENTRAÎNEMENT ")
    logger.info("="*40)

    # 1. Préparer les données historiques complètes
    all_data_dict = prepare_historical_data(config, apis)

    if not all_data_dict:
        logger.error("Aucune donnée historique préparée pour l'entraînement. Pipeline arrêté.")
        return

    # Concaténer les données de tous les actifs pour l'entraînement global
    combined_data = pd.concat(list(all_data_dict.values()))
    # Mélanger les données de différents actifs pour un entraînement plus robuste
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True) # Mélanger

    if combined_data.empty:
        logger.error("DataFrame combiné vide après concaténation. Pipeline arrêté.")
        return

    logger.info(f"Données combinées pour l'entraînement: {len(combined_data)} lignes.")

    # Séparer features et cibles
    # Les colonnes 'close' est le prix brut, 'discrete_label' pour XGBoost, 'future_price_ratio_target' pour FQ
    # Toutes les autres colonnes sont des features
    # Déterminer all_potential_features: toutes les colonnes sauf les cibles et prix bruts non transformés
    all_potential_features = [col for col in combined_data.columns if col not in ['open', 'high', 'low', 'close', 'discrete_label', 'future_price_ratio_target']]
    
    # S'assurer qu'il n'y a pas de colonnes non-numériques ou problématiques restantes par erreur
    potential_issues = []
    for col in all_potential_features:
        if combined_data[col].dtype == 'object' or combined_data[col].dtype == 'bool':
            try:
                # Tenter une conversion, si elle échoue, c'est un problème
                pd.to_numeric(combined_data[col])
            except ValueError:
                potential_issues.append(col)
    
    if potential_issues:
        logger.warning(f"Colonnes potentiellement non-numériques ou booléennes dans les features: {potential_issues}. Elles pourraient causer des erreurs ou être mal interprétées.")
        # Option: Supprimer ces colonnes ou les convertir explicitement si la logique est connue
        # all_potential_features = [col for col in all_potential_features if col not in potential_issues]


    X_all = combined_data[all_potential_features]
    y_xgb = combined_data['discrete_label']
    y_fq_ratio_target = combined_data['future_price_ratio_target'].values.reshape(-1, 1) # FQ cible est le ratio futur

    # 2. Entraîner les Scalers
    # Utiliser TOUTES les features (sauf les cibles et prix bruts) pour entraîner le scaler.
    # Le scaler sera utilisé pour normaliser les features de XGBoost et les séquences de FQ.
    feature_engineer.fit_and_save_scalers(X_all)
    
    # Charger le scaler entraîné
    scaler = feature_engineer.load_feature_scaler()
    
    if scaler is None:
        logger.error("Échec du chargement du scaler. Impossible de poursuivre l'entraînement.")
        return

    # 3. Créer les sets d'entraînement pour XGBoost (features vectorielles)
    # S'assurer que y_xgb a suffisamment de classes pour stratify (au moins 2 instances par classe)
    if y_xgb.nunique() < 2:
        logger.warning(f"Moins de 2 classes uniques dans y_xgb ({y_xgb.unique()}). Stratification désactivée pour train_test_split.")
        X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(
            X_all, y_xgb, test_size=0.2, random_state=42
        )
    else:
        try:
            X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(
                X_all, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
            )
        except ValueError as e:
            logger.warning(f"Erreur de stratification pour XGBoost ({e}). Tentative sans stratification.")
            X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(
                X_all, y_xgb, test_size=0.2, random_state=42
            )

    
    # Appliquer le scaling
    X_xgb_train_scaled = pd.DataFrame(
        scaler.transform(X_xgb_train),
        columns=X_xgb_train.columns,
        index=X_xgb_train.index
    )
    
    X_xgb_test_scaled = pd.DataFrame(
        scaler.transform(X_xgb_test),
        columns=X_xgb_test.columns,
        index=X_xgb_test.index
    )
    
    # 4. Créer les sets d'entraînement pour FutureQuant (séquences)
    # Paramètres pour les séquences FQ
    fq_window_size_in = config.getint('futurequant', 'window_size_in', fallback=32)
    fq_window_size_out = config.getint('futurequant', 'window_size_out', fallback=8) # S'assurer que c'est bien le bon paramètre

    # Préparer les listes pour collecter les séquences et leurs cibles
    X_fq_seq_list = []
    y_fq_ratio_list = []

    # Pour chaque asset, créer des séquences en parcourant la série temporelle
    for symbol_market, data_df_original_index in all_data_dict.items():
        # S'assurer que les colonnes de features existent dans les données de l'actif
        # et qu'elles sont dans le même ordre que all_potential_features
        # Il est crucial que data_df_original_index contienne bien all_potential_features
        
        # Filtrer data_df_original_index pour ne garder que les all_potential_features
        # et gérer les colonnes manquantes si nécessaire (bien que prepare_historical_data devrait les fournir)
        asset_features_df = data_df_original_index.reindex(columns=all_potential_features).fillna(0.0)

        if scaler:
            data_df_scaled_values = scaler.transform(asset_features_df)
            data_df_scaled = pd.DataFrame(
                data_df_scaled_values,
                index=asset_features_df.index, 
                columns=all_potential_features
            )
        else:
            # Ceci ne devrait pas arriver si le scaler a été chargé correctement
            data_df_scaled = asset_features_df

        # Obtenir la série des cibles pour cet actif
        fq_target_series = data_df_original_index['future_price_ratio_target']

        # Itérer dans la série temporelle pour créer des séquences
        for i in range(fq_window_size_in, len(data_df_scaled) - fq_window_size_out + 1):
            # Séquence d'entrée: fenêtre de features scalées
            input_seq_df = data_df_scaled.iloc[i-fq_window_size_in:i]
            
            # Vérifier que la séquence a la bonne longueur
            if len(input_seq_df) != fq_window_size_in:
                logger.warning(f"Séquence FQ pour {symbol_market} à l'index {i} a une longueur incorrecte ({len(input_seq_df)} au lieu de {fq_window_size_in}). Ignorée.")
                continue

            input_seq = input_seq_df.values
            
            # Cible: ratio de prix futur à i+fq_window_size_out
            # L'indexation doit être prudente ici. fq_target_series est alignée avec data_df_original_index
            # Si data_df_scaled a le même index, alors iloc[i+fq_window_size_out-1] sur fq_target_series est correct.
            # L'index 'i' ici est pour data_df_scaled.
            target_timestamp = data_df_scaled.index[i+fq_window_size_out-1] # Timestamp de la cible
            
            try:
                target_ratio = fq_target_series.loc[target_timestamp]
            except KeyError:
                logger.warning(f"Timestamp cible {target_timestamp} non trouvé dans fq_target_series pour {symbol_market}. Séquence ignorée.")
                continue

            # Ignorer si la cible est NaN (peut arriver si fq_window_size_out est grand)
            if np.isnan(target_ratio):
                # logger.debug(f"Cible NaN pour FQ à {target_timestamp} pour {symbol_market}. Séquence ignorée.")
                continue
                
            # Ajouter à nos collections
            X_fq_seq_list.append(input_seq)
            y_fq_ratio_list.append(target_ratio)

    if not X_fq_seq_list:
        logger.error("Aucune séquence FQ n'a pu être générée. Vérifiez les données et les paramètres de fenêtre.")
        # On pourrait arrêter ici ou continuer sans FQ. Pour l'instant, on continue.
        X_fq_train_seq = np.array([])
        y_fq_train_ratio = np.array([])
    else:
        X_fq_train_seq = np.array(X_fq_seq_list)
        y_fq_train_ratio = np.array(y_fq_ratio_list).reshape(-1, 1)

    logger.info(f"Séquences FQ créées. X_fq_train_seq forme: {X_fq_train_seq.shape}, y_fq_train_ratio forme: {y_fq_train_ratio.shape}")

    # 5. Entraîner les Modèles
    # Entraîner XGBoost
    logger.info("Début entraînement modèle XGBoost...")
    model_xgboost.train_xgboost(X_xgb_train_scaled, y_xgb_train, X_xgb_test_scaled, y_xgb_test)
    
    # Entraîner FutureQuant (seulement si des séquences ont été créées)
    if X_fq_train_seq.shape[0] > 0:
        logger.info("Début entraînement modèle FutureQuant...")
        # Définir les quantiles à prédire (ceux de la config FutureQuant)
        quantiles_str = config.get('futurequant', 'quantiles', fallback='0.1,0.5,0.9')
        quantiles = [float(q.strip()) for q in quantiles_str.split(',')]
        
        # La fonction train_futurequant de model_futurequant.py utilise config pour les quantiles
        # donc on n'a pas besoin de les passer explicitement ici si c'est bien géré dans le module.
        # Vérifions: train_futurequant(X_train_seq, y_train_ratio, config)
        # Dans model_futurequant.py: def train_futurequant_model(X_train_seq: np.ndarray, y_train_ratio: np.ndarray, config: ConfigParser) -> Optional[keras.Model]:
        # Et elle récupère les quantiles de la config.
        # Donc, le `quantiles` variable ici est redondant si on passe la config.
        # La fonction train_futurequant a été modifiée dans le résumé pour être `model_futurequant.train_futurequant_model(X_fq_train_seq, y_fq_train_ratio, config) # Utiliser la fonction du module, elle prend la config
        model_futurequant.train_futurequant_model(X_fq_train_seq, y_fq_train_ratio, config) # Utiliser la fonction du module, elle prend la config
    else:
        logger.warning("Aucune séquence FQ pour entraîner le modèle FutureQuant. Étape sautée.")


    # 6. Sauvegarder les paramètres d'entraînement
    # Récupérer les quantiles depuis la config FutureQuant pour la sauvegarde
    quantiles_str_final = config.get('futurequant', 'quantiles', fallback='0.1,0.5,0.9')
    quantiles_final = [float(q.strip()) for q in quantiles_str_final.split(',')]

    training_params = {
        'all_potential_features': all_potential_features, # Ajouté
        'xgboost_features': X_xgb_train_scaled.columns.tolist(), # Utiliser les colonnes réelles de X_xgb_train_scaled
        'futurequant_quantiles': quantiles_final, # Utiliser les quantiles de la config FQ
        'date_trained': datetime.now().isoformat(),
        'training_size': len(X_all),
        'window_size_in': fq_window_size_in,
        'window_size_out': fq_window_size_out
    }
    
    training_params_file = config.get('training', 'training_params_file', fallback='training_params.json')
    
    try:
        with open(training_params_file, 'w') as f:
            json.dump(training_params, f, indent=4)
        logger.info(f"Paramètres d'entraînement sauvegardés dans {training_params_file}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des paramètres d'entraînement: {e}")

    logger.info("="*40)
    logger.info(" FIN DU PIPELINE D'ENTRAÎNEMENT ")
    logger.info("="*40)

    # Retourner un message de succès
    return True

def load_prepared_data_for_analysis(config: ConfigParser) -> Dict[str, pd.DataFrame]:
    """
    Charge les données préparées pour l'analyse de backtest.
    
    Args:
        config: Configuration du système
        
    Returns:
        Dictionnaire d'actifs avec leurs DataFrames de données préparées
    """
    logger.info("Chargement des données préparées pour l'analyse...")
    
    # Récupérer les paramètres de configuration
    data_dir = config.get('paths', 'data_dir', fallback='data')
    prepared_data_file = config.get('training', 'prepared_data_file', fallback='prepared_data.pickle')
    prepared_data_path = os.path.join(data_dir, prepared_data_file)
    
    # Vérifier si le fichier existe
    if not os.path.exists(prepared_data_path):
        logger.warning(f"Fichier de données préparées '{prepared_data_path}' non trouvé.")
        return {}
    
    # Charger les données depuis le fichier pickle
    try:
        with open(prepared_data_path, 'rb') as f:
            prepared_data = pickle.load(f)
        
        logger.info(f"Données préparées chargées depuis {prepared_data_path}")
        return prepared_data
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données préparées: {str(e)}")
        return {}

def is_training_needed(config: ConfigParser) -> bool:
    """
    Vérifie si un ré-entraînement est nécessaire en fonction de la date du dernier entraînement
    et de la configuration.
    
    Args:
        config: Configuration du système
        
    Returns:
        Boolean indiquant si un entraînement est nécessaire
    """
    # Vérifier si les fichiers de modèles existent
    training_params_file = config.get('training', 'training_params_file', fallback='training_params.json')
    
    if not os.path.exists(training_params_file):
        logger.info("Fichier de paramètres d'entraînement introuvable. Entraînement nécessaire.")
        return True

    # Vérifier la date du dernier entraînement
    try:
        with open(training_params_file, 'r') as f:
            params = json.load(f)
            
        if 'date_trained' in params:
            date_trained = datetime.fromisoformat(params['date_trained'])
            days_since_training = (datetime.now() - date_trained).days
            
            # Vérifier si le nombre de jours depuis le dernier entraînement dépasse le seuil
            retraining_days = config.getint('training', 'retraining_days', fallback=30)
            if days_since_training > retraining_days:
                logger.info(f"Dernier entraînement il y a {days_since_training} jours. Ré-entraînement nécessaire.")
                return True
            else:
                logger.info(f"Dernier entraînement il y a {days_since_training} jours. Ré-entraînement non nécessaire.")
                return False
        else:
            logger.warning("Date d'entraînement non trouvée dans les paramètres.")
            return True
            
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de la date d'entraînement: {e}")
        return True

if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training_pipeline.log"),
            logging.StreamHandler()
        ]
    )
    
    # Charger la configuration
    config = ConfigParser()
    config.read('config.ini')
    
    # Vérifier si un ré-entraînement est nécessaire
    if is_training_needed(config):
        # Exécuter le pipeline d'entraînement
        run_training_pipeline(config, {})
    else:
        logger.info("Ré-entraînement non nécessaire. Pipeline terminé.") 