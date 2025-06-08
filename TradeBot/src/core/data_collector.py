import requests
import pandas as pd
import numpy as np
import logging
import os
import sys
import time
from configparser import ConfigParser, NoSectionError, NoOptionError
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List

# Importer les bibliothèques API dédiées
import yfinance as yf
try:
    import fredapi
    from alpha_vantage.timeseries import TimeSeries  # Pour actions/crypto/forex
    from alpha_vantage.economic import Economic  # Pour indicateurs éco
    from alpha_vantage.commodities import Commodities  # Pour matières premières
except ImportError:
    logging.warning("Certaines bibliothèques d'API financières non disponibles. Installer via requirements.txt.")

# Importer le module de gestion de base de données
import db_handler


# --- Configuration Globale & Logging ---
CONFIG_FILE = 'config.ini'
config = ConfigParser()
logger = logging.getLogger(__name__)

# --- Helper pour la configuration ---
def get_config_option(section: str, option: str, fallback=None):
    """Récupère une option de configuration avec gestion des erreurs de section/option."""
    try:
        return config.get(section, option)
    except (NoSectionError, NoOptionError):
        if fallback is not None:
            # logger.warning(f"Section '{section}' ou option '{option}' manquante dans config.ini. Utilise fallback: {fallback}")
            return fallback
        else:
             logger.error(f"Section '{section}' ou option '{option}' manquante dans config.ini et aucun fallback fourni.")
             raise # Renvoyer l'erreur si pas de fallback et option requise

def get_config_list(section: str, option: str, fallback: List = []) -> List[str]:
    """Récupère une option de configuration qui est une liste séparée par des virgules."""
    value = get_config_option(section, option, fallback=', '.join(fallback))
    if isinstance(value, str):
         return [item.strip() for item in value.split(',') if item.strip()]
    return fallback

# --- Configuration Logging ---
def setup_logging(config):
    """Configure le logging pour le script de collecte."""
    try:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        if root_logger.hasHandlers():
             root_logger.handlers.clear()

        log_file = get_config_option('logging', 'log_file', fallback='data_collector.log') # Chemin différent pour le collecteur
        log_level_str = get_config_option('logging', 'log_level', fallback='INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)

        logger.info("Logging configuré pour le data collector.")

    except Exception as e:
        print(f"ERREUR: Impossible de configurer le logging depuis {CONFIG_FILE}: {e}")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
        logger.warning("Utilisation de la configuration de logging par défaut.")


# --- Fonctions de Collecte par Source ---

def fetch_yahoo_finance(symbol: str, start_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """Récupère les données historiques (quotidiennes) depuis Yahoo Finance."""
    logger.info(f"Fetching Yahoo Finance data for {symbol}...")
    try:
        # yfinance gère les dates start/end directement en datetime
        # Si start_date est None, yfinance récupère tout l'historique disponible par défaut
        # Si start_date est fourni, yfinance récupère à partir de cette date
        # End date est implicitement aujourd'hui
        data = yf.download(symbol, start=start_date, end=datetime.now(timezone.utc), interval="1d", progress=False)
        if data.empty:
            logger.warning(f"Aucune donnée retournée par yfinance pour {symbol} depuis {start_date}.")
            return None

        # Nettoyer les noms de colonnes et s'assurer de la présence des colonnes clés
        data.columns = [col.replace(' ', '_').lower() for col in data.columns]
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close'] # 'adj_close' est important
        if not all(col in data.columns for col in required_cols):
             logger.warning(f"Certaines colonnes requises manquantes pour {symbol} dans yfinance: {set(required_cols) - set(data.columns)}")
             # Continuer même si des colonnes manquent, mais log warning

        # Utiliser l'index (datetime) comme colonne 'timestamp'
        data.reset_index(inplace=True)
        data.rename(columns={'date': 'timestamp'}, inplace=True) # yfinance utilise 'Date' comme nom d'index

        # S'assurer que le timestamp est timezone-aware (yfinance peut retourner des indices sans tz)
        if data['timestamp'].dtype == 'datetime64[ns]':
             data['timestamp'] = data['timestamp'].dt.tz_localize(timezone.utc) # Assumer UTC si non spécifié

        logger.info(f"Yahoo Finance data fetched for {symbol} ({len(data)} rows).")
        return data

    except Exception as e:
        logger.error(f"Error fetching yfinance data for {symbol}: {e}", exc_info=True)
        return None

def fetch_fred_series(series_id: str, api_key: str, start_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """Récupère les données d'une série FRED."""
    if not api_key:
        logger.error("Clé API FRED non fournie.")
        return None

    logger.info(f"Fetching FRED series {series_id}...")
    try:
        # fredapi gère les dates start/end et la clé API
        # fred = fredapi.Fred(api_key=api_key) # Ne pas créer une nouvelle instance à chaque appel, ou la mettre en cache
        # Let's create it here for simplicity for now.
        fred = fredapi.Fred(api_key=api_key)

        # Convertir start_date en string YYYY-MM-DD si présent
        start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None

        data = fred.get_series_observations(series_id, observation_start=start_date_str)

        if data is None or data.empty:
            logger.warning(f"Aucune donnée retournée par FRED pour {series_id} depuis {start_date_str}.")
            return None

        # fredapi retourne une Series pandas avec DatetimeIndex. Convertir en DataFrame.
        df = data.reset_index()
        df.rename(columns={'index': 'timestamp', 0: 'value'}, inplace=True) # Nom par défaut est 0

        # Assurer le format timestamp timezone-aware (fredapi index est souvent sans tz)
        if df['timestamp'].dtype == 'datetime64[ns]':
             df['timestamp'] = df['timestamp'].dt.tz_localize(timezone.utc) # Assumer UTC

        # Ajouter une colonne identifiant la série
        df['series_id'] = series_id

        logger.info(f"FRED series {series_id} data fetched ({len(df)} rows).")
        return df

    except Exception as e:
        logger.error(f"Error fetching FRED series {series_id}: {e}", exc_info=True)
        return None


def fetch_alpha_vantage_series(symbol: str, data_type: str, api_key: str, start_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """Récupère différentes séries depuis Alpha Vantage (actions, crypto, forex, commodities, eco)."""
    if not api_key:
        logger.error("Clé API Alpha Vantage non fournie.")
        return None

    logger.info(f"Fetching Alpha Vantage data for {symbol}, type {data_type}...")

    # Gérer les limites de rate (5 appels/min pour l'API gratuite)
    # Une approche simple est d'ajouter un délai fixe. Une approche plus sophistiquée tiendrait compte du temps passé.
    # On gérera le délai globalement dans la boucle principale.

    data = None
    try:
        # --- Financial Time Series (Stocks, Crypto, Forex) ---
        if data_type in ['TIME_SERIES_DAILY_ADJUSTED', 'TIME_SERIES_WEEKLY_ADJUSTED', 'TIME_SERIES_MONTHLY_ADJUSTED']:
            # symbol est le ticker (ex: AAPL)
            ts = TimeSeries(key=api_key, output_format='pandas')
            data, meta_data = ts.get_time_series(symbol=symbol, interval=data_type.replace('TIME_SERIES_', '').lower(), outputsize='full') # 'full' pour tout l'historique

            # Alpha Vantage retourne un MultiIndex ou DatetimeIndex, noms de colonnes standardisés avec '1. open' etc.
            if data is not None:
                 data.columns = [col.split('. ')[1].replace(' ', '_') for col in data.columns] # Nettoyer noms colonnes
                 # L'index est un DatetimeIndex, souvent sans timezone. Convertir en UTC.
                 if data.index.tzinfo is None:
                      data = data.tz_localize(timezone.utc)
                 data.reset_index(inplace=True)
                 data.rename(columns={'date': 'timestamp'}, inplace=True)
                 logger.info(f"AV {data_type} data fetched for {symbol} ({len(data)} rows).")

        elif data_type in ['CRYPTO_DAILY', 'CRYPTO_WEEKLY', 'CRYPTO_MONTHLY']:
             # symbol est le symbole crypto (ex: BTC), la quote currency doit être spécifiée.
             # La fonction get_crypto_data nécessite la paire (ex: BTC/USD).
             # On doit la construire si le config ne fournit que le symbole base.
             # Utilisons la convention de config pour construire la paire si nécessaire.
             quote_currency = config.get('collector.alpha_vantage', 'forex_quote_currency', fallback='USD') # Réutiliser la config forex quote
             symbol_pair = f"{symbol}/{quote_currency}" # Ex: BTC/USD
             interval_map = {'CRYPTO_DAILY': 'daily', 'CRYPTO_WEEKLY': 'weekly', 'CRYPTO_MONTHLY': 'monthly'}
             ts = TimeSeries(key=api_key, output_format='pandas')
             # Note: get_crypto_data peut ne pas avoir toutes les options de get_time_series
             data, meta_data = ts.get_crypto_data(symbol=symbol, market=quote_currency, extract_data=True, date_type=interval_map.get(data_type, 'daily')) # outputsize='full' n'existe pas ici?

             if data is not None:
                  # Les colonnes sont déjà des noms propres (open, high, etc.)
                  # L'index est un DatetimeIndex, souvent sans timezone. Convertir en UTC.
                  if data.index.tzinfo is None:
                       data = data.tz_localize(timezone.utc)
                  data.reset_index(inplace=True)
                  data.rename(columns={'date': 'timestamp'}, inplace=True)
                  # Ajouter le symbole original et la paire pour le stockage/traçage
                  data['symbol'] = symbol
                  data['market_quote'] = quote_currency # ou la paire complète
                  logger.info(f"AV {data_type} data fetched for {symbol_pair} ({len(data)} rows).")

        elif data_type in ['FOREX_DAILY', 'FOREX_WEEKLY', 'FOREX_MONTHLY']:
            # symbol est la base currency (ex: EUR), la quote currency doit être spécifiée.
            base_currency = symbol
            quote_currency = config.get('collector.alpha_vantage', 'forex_quote_currency', fallback='USD')
            symbol_pair = f"{base_currency}{quote_currency}" # Ex: EURUSD (souvent sans /)
            interval_map = {'FOREX_DAILY': 'daily', 'FOREX_WEEKLY': 'weekly', 'FOREX_MONTHLY': 'monthly'}
            ts = TimeSeries(key=api_key, output_format='pandas')
            data, meta_data = ts.get_forex_data(from_symbol=base_currency, to_symbol=quote_currency, interval=interval_map.get(data_type, 'daily'), outputsize='full')

            if data is not None:
                 data.columns = [col.split('. ')[1].replace(' ', '_') for col in data.columns]
                 if data.index.tzinfo is None:
                       data = data.tz_localize(timezone.utc)
                 data.reset_index(inplace=True)
                 data.rename(columns={'date': 'timestamp'}, inplace=True)
                 data['symbol'] = symbol_pair # Stocker la paire complète comme identifiant
                 logger.info(f"AV {data_type} data fetched for {symbol_pair} ({len(data)} rows).")

        # --- Economic Indicators ---
        elif data_type == 'ECONOMIC_INDICATORS':
            # symbol est le nom de l'indicateur selon AV (ex: CPI, GDP, UNEMPLOYMENT)
            eco = Economic(key=api_key, output_format='pandas')
            # La méthode spécifique dépend de l'indicateur (get_retail_sales, get_treasury_yield, etc.)
            # Il faudrait une map ou une logique par indicateur.
            # Pour simplifier, on va utiliser get_economic_indicator et le nom comme paramètre 'indicator'.
            # Note: La documentation AV peut être nécessaire pour les noms exacts et les méthodes.
            # Let's assume the 'symbol' provided in config matches AV indicator names.
            try:
                 data, meta_data = eco.get_economic_indicator(indicator=symbol) # Assumes 'symbol' is the indicator name for this function
                 if data is not None:
                      # Les colonnes sont souvent ['date', 'value']. L'index est date.
                      if data.index.tzinfo is None:
                          data = data.tz_localize(timezone.utc)
                      data.reset_index(inplace=True)
                      data.rename(columns={'date': 'timestamp'}, inplace=True)
                      data['indicator_name'] = symbol # Ajouter le nom de l'indicateur
                      logger.info(f"AV Economic Indicator {symbol} data fetched ({len(data)} rows).")
            except AttributeError: # Si get_economic_indicator n'existe pas ou n'est pas la bonne fonction
                 logger.error(f"Alpha Vantage API method for economic indicator '{symbol}' not found or not implemented in library.")
                 return None


        # --- Commodities ---
        elif data_type == 'COMMODITIES':
             # symbol est le nom de la commodité selon AV (ex: WTI, BRENT, NATURAL_GAS)
             comm = Commodities(key=api_key, output_format='pandas')
             # Comme pour les indicateurs éco, la méthode dépend de la commodité.
             # Utilisons get_commodity et le nom comme paramètre 'commodity'.
             try:
                  data, meta_data = comm.get_commodity(commodity=symbol)
                  if data is not None:
                       if data.index.tzinfo is None:
                           data = data.tz_localize(timezone.utc)
                       data.reset_index(inplace=True)
                       data.rename(columns={'date': 'timestamp'}, inplace=True)
                       data['commodity_name'] = symbol
                       logger.info(f"AV Commodity {symbol} data fetched ({len(data)} rows).")
             except AttributeError:
                  logger.error(f"Alpha Vantage API method for commodity '{symbol}' not found or not implemented in library.")
                  return None


        else:
            logger.error(f"Data type AV non supporté: {data_type} pour {symbol}.")
            return None


        # --- Filtrer par date si nécessaire ---
        if data is not None and start_date is not None:
             # Assurer que start_date est timezone-aware pour la comparaison
             if start_date.tzinfo is None:
                 start_date = start_date.replace(tzinfo=timezone.utc)

             initial_rows = len(data)
             data = data[data['timestamp'] >= start_date]
             if len(data) < initial_rows:
                  logger.info(f"Filtré les données AV pour {symbol} à partir de {start_date} ({initial_rows} -> {len(data)} lignes).")


        return data

    except Exception as e:
        logger.error(f"Error fetching Alpha Vantage data for {symbol}, type {data_type}: {e}", exc_info=True)
        # Gérer spécifiquement les RateLimitExceeded si possible
        if "limit" in str(e).lower() or "rate" in str(e).lower():
             logger.warning("Alpha Vantage rate limit hit. Consider adding delays or upgrading API key.")
             # Le gestionnaire global gérera l'attente.
             # On peut renvoyer un signal pour indiquer un rate limit au gestionnaire principal.
             # Pour l'instant, on retourne juste None et on log.
        return None


# --- Gestion des Délais pour APIs (Rudimentaire) ---
_last_av_call_time = 0
_av_call_count_in_minute = 0
_av_max_calls_per_minute = 5 # Limit for free API

def wait_for_alpha_vantage_rate_limit():
    """Pause execution to respect Alpha Vantage rate limits."""
    global _last_av_call_time, _av_call_count_in_minute

    current_time = time.time()
    time_since_last_call = current_time - _last_av_call_time

    # Reset count if a minute has passed
    if time_since_last_call > 60:
        _av_call_count_in_minute = 0

    _av_call_count_in_minute += 1
    _last_av_call_time = current_time

    # If we are over the limit in the current minute, wait until the minute resets
    if _av_call_count_in_minute > _av_max_calls_per_minute:
        wait_until = current_time + (60 - time_since_last_call) # Wait for the rest of the minute
        sleep_duration = max(1, wait_until - time.time()) # Ensure at least 1 second sleep
        logger.warning(f"Approaching AV rate limit ({_av_max_calls_per_minute} calls/min). Sleeping for {sleep_duration:.2f} seconds.")
        time.sleep(sleep_duration)
        _av_call_count_in_minute = 0 # Reset count after waiting for the minute


# --- Main Collector Logic ---
def main():
    """Fonction principale pour la collecte de données."""
    # Charger la configuration
    if not config.read(CONFIG_FILE):
        print(f"ERREUR: Fichier de configuration '{CONFIG_FILE}' non trouvé ou vide.")
        sys.exit(1)

    # Configurer le logging
    setup_logging(config)
    logger.info("="*40)
    logger.info(" DÉMARRAGE DU DATA COLLECTOR ")
    logger.info("="*40)

    # Obtenir l'engine de base de données
    engine = db_handler.get_db_engine(config)
    if not engine:
        logger.critical("Échec de l'initialisation de la base de données. Arrêt du collecteur.")
        sys.exit(1)

    # --- Collecte Yahoo Finance ---
    if 'yahoo_finance' in get_config_list('collector', 'enabled_sources'):
        logger.info("\n--- COLLECTE YAHOO FINANCE ---")
        yf_symbols = get_config_list('collector.yahoo_finance', 'symbols')
        for symbol in yf_symbols:
            table_name = f"yf_{symbol.replace('^', '').replace('=', '_').replace('-', '_')}".lower()
            latest_ts = db_handler.get_latest_timestamp(engine, table_name, 'timestamp')

            # Récupérer à partir du dernier timestamp + 1 jour pour éviter les doublons (pour les données journalières)
            # Yahoo Finance API pourrait retourner des données pour le jour de start_date si elle est en milieu de journée, potentiellement créer des doublons.
            # Alternative plus robuste: récupérer la dernière ligne de la DB, et fetcher à partir de la date *suivante*.
            # Pour yfinance, si start=X, il inclut X. Donc on veut start=latest_ts + 1 jour.
            fetch_start_date = latest_ts + timedelta(days=1) if latest_ts else None
            logger.info(f"Dernier timestamp connu pour {symbol} ({table_name}): {latest_ts}. Fetching from {fetch_start_date}.")

            data = fetch_yahoo_finance(symbol, fetch_start_date)
            if data is not None and not data.empty:
                # Si on fetch à partir de latest_ts + 1 jour, on peut normalement append sans vérifier doublons (pour 1d)
                # Si on ne peut pas garantir fetch_start_date exclusif, il faudrait gérer les doublons (ex: drop_duplicates basé sur timestamp)
                # ou utiliser if_exists='append' avec une contrainte UNIQUE sur (timestamp, symbol) dans la DB.
                # Assumons ici que fetch_start_date + 1 jour fonctionne pour éviter les doublons sur 1d.
                db_handler.store_dataframe(engine, data, table_name, if_exists='append')
            else:
                 logger.warning(f"Aucune donnée Yahoo Finance récente à stocker pour {symbol}.")


    # --- Collecte FRED ---
    if 'fred' in get_config_list('collector', 'enabled_sources'):
        logger.info("\n--- COLLECTE FRED ---")
        fred_api_key = get_config_option('api_keys', 'fred_api_key', fallback=None)
        if not fred_api_key:
            logger.warning("FRED API Key non configurée. Saut de la collecte FRED.")
        else:
            fred_series_ids = get_config_list('collector.fred', 'series_ids')
            for series_id in fred_series_ids:
                table_name = f"fred_{series_id}".lower()
                latest_ts = db_handler.get_latest_timestamp(engine, table_name, 'timestamp')

                # FRED API get_series_observations avec observation_start inclut la date de début.
                # On veut donc fetcher à partir de la date *suivante* au dernier timestamp connu.
                fetch_start_date = latest_ts + timedelta(days=1) if latest_ts else None # Ajouter 1 jour pour FRED aussi

                logger.info(f"Dernier timestamp connu pour FRED {series_id} ({table_name}): {latest_ts}. Fetching from {fetch_start_date}.")

                data = fetch_fred_series(series_id, fred_api_key, fetch_start_date)
                if data is not None and not data.empty:
                    db_handler.store_dataframe(engine, data, table_name, if_exists='append')
                else:
                    logger.warning(f"Aucune donnée FRED récente à stocker pour {series_id}.")


    # --- Collecte Alpha Vantage ---
    if 'alpha_vantage' in get_config_list('collector', 'enabled_sources'):
        logger.info("\n--- COLLECTE ALPHA VANTAGE ---")
        av_api_key = get_config_option('collector.alpha_vantage.api', 'api_key', fallback=None)
        if not av_api_key:
            logger.warning("Alpha Vantage API Key non configurée dans [collector.alpha_vantage.api]. Saut de la collecte Alpha Vantage.")
        else:
            # Collecte Actions (TIME_SERIES_DAILY_ADJUSTED, etc.)
            av_stock_symbols = get_config_list('collector.alpha_vantage', 'stocks')
            if av_stock_symbols:
                 logger.info("--- AV Stocks ---")
                 for symbol in av_stock_symbols:
                     wait_for_alpha_vantage_rate_limit() # Respecter le rate limit
                     table_name = f"av_stock_daily_{symbol}".lower() # Exemple: stock_daily_aapl
                     latest_ts = db_handler.get_latest_timestamp(engine, table_name, 'timestamp')
                     # AV get_time_series('full') donne tout l'historique. On peut le fetch une fois et le mettre à jour?
                     # L'alternative 'compact' donne 100 dernières bougies. Pas idéal pour la mise à jour incrémentale depuis n'importe quelle date.
                     # Le plus simple avec 'full' est de fetcher tout et de filtrer/ajouter seulement les nouvelles données.
                     # Ou mieux: utiliser 'compact' pour la mise à jour récente, et 'full' pour la première fois.
                     # Pour cette esquisse, fetchons 'full' et filtrons après.
                     data = fetch_alpha_vantage_series(symbol, 'TIME_SERIES_DAILY_ADJUSTED', av_api_key) # outputsize='full' dans la fonction

                     if data is not None and not data.empty:
                         if latest_ts:
                              # Filtrer les données déjà présentes
                              data_to_store = data[data['timestamp'] > latest_ts].copy() # Fetché 'full', on ne stocke que les NOUVELLES
                              logger.info(f"Filtré données AV pour {symbol} pour ajouter après {latest_ts} ({len(data)} -> {len(data_to_store)} lignes).")
                         else:
                              data_to_store = data.copy() # Pas de données existantes, stocker tout

                         if not data_to_store.empty:
                              db_handler.store_dataframe(engine, data_to_store, table_name, if_exists='append')
                         else:
                              logger.warning(f"Aucune donnée AV récente à stocker pour {symbol}.")
                     else:
                          logger.warning(f"Aucune donnée AV (Stock) retournée ou erreur pour {symbol}.")


            # Collecte Crypto (CRYPTO_DAILY, etc.)
            av_crypto_symbols = get_config_list('collector.alpha_vantage', 'crypto')
            if av_crypto_symbols:
                 logger.info("--- AV Crypto ---")
                 for symbol in av_crypto_symbols:
                     wait_for_alpha_vantage_rate_limit()
                     # Note: AV Crypto nécessite la paire (ex: BTC/USD). Notre config a base+quote.
                     # La fonction fetch_alpha_vantage_series gère la construction de la paire.
                     # La table sera nommée d'après le symbole base et la quote.
                     quote_currency = get_config_option('collector.alpha_vantage', 'forex_quote_currency', fallback='USD')
                     symbol_pair_for_table = f"{symbol}_{quote_currency}".lower()
                     table_name = f"av_crypto_daily_{symbol_pair_for_table}" # Exemple: crypto_daily_btc_usd

                     latest_ts = db_handler.get_latest_timestamp(engine, table_name, 'timestamp')

                     # AV Crypto get_crypto_data a un paramètre date_type ('daily', 'weekly', 'monthly')
                     # Il faut spécifier 'daily' pour récupérer l'historique quotidien.
                     data = fetch_alpha_vantage_series(symbol, 'CRYPTO_DAILY', av_api_key) # Pass base symbol, type is daily

                     if data is not None and not data.empty:
                         if latest_ts:
                              data_to_store = data[data['timestamp'] > latest_ts].copy()
                              logger.info(f"Filtré données AV pour {symbol} pour ajouter après {latest_ts} ({len(data)} -> {len(data_to_store)} lignes).")
                         else:
                              data_to_store = data.copy()

                         if not data_to_store.empty:
                              db_handler.store_dataframe(engine, data_to_store, table_name, if_exists='append')
                         else:
                              logger.warning(f"Aucune donnée AV récente à stocker pour {symbol}.")
                     else:
                          logger.warning(f"Aucune donnée AV (Crypto) retournée ou erreur pour {symbol}.")


            # Collecte Forex (FOREX_DAILY, etc.)
            av_forex_symbols = get_config_list('collector.alpha_vantage', 'forex')
            if av_forex_symbols:
                 logger.info("--- AV Forex ---")
                 quote_currency = get_config_option('collector.alpha_vantage', 'forex_quote_currency', fallback='USD')
                 for symbol in av_forex_symbols: # symbol ici est la base currency
                     wait_for_alpha_vantage_rate_limit()
                     symbol_pair_for_table = f"{symbol}_{quote_currency}".lower() # Ex: eur_usd
                     table_name = f"av_forex_daily_{symbol_pair_for_table}"

                     latest_ts = db_handler.get_latest_timestamp(engine, table_name, 'timestamp')
                     data = fetch_alpha_vantage_series(symbol, 'FOREX_DAILY', av_api_key) # Pass base symbol, type is daily

                     if data is not None and not data.empty:
                         if latest_ts:
                              data_to_store = data[data['timestamp'] > latest_ts].copy()
                              logger.info(f"Filtré données AV pour {symbol} pour ajouter après {latest_ts} ({len(data)} -> {len(data_to_store)} lignes).")
                         else:
                              data_to_store = data.copy()

                         if not data_to_store.empty:
                              db_handler.store_dataframe(engine, data_to_store, table_name, if_exists='append')
                         else:
                              logger.warning(f"Aucune donnée AV récente à stocker pour {symbol}.")
                     else:
                          logger.warning(f"Aucune donnée AV (Forex) retournée ou erreur pour {symbol}.")


            # Collecte Indicateurs Économiques (ECONOMIC_INDICATORS)
            av_eco_indicators = get_config_list('collector.alpha_vantage', 'economic_indicators')
            if av_eco_indicators:
                 logger.info("--- AV Economic Indicators ---")
                 for indicator_name in av_eco_indicators: # symbol ici est le nom de l'indicateur AV
                     wait_for_alpha_vantage_rate_limit()
                     table_name = f"av_eco_{indicator_name}".lower()

                     latest_ts = db_handler.get_latest_timestamp(engine, table_name, 'timestamp')
                     # Pour les indicateurs éco, la fonction fetch_alpha_vantage_series prend le nom de l'indicateur comme 'symbol'
                     data = fetch_alpha_vantage_series(indicator_name, 'ECONOMIC_INDICATORS', av_api_key)

                     if data is not None and not data.empty:
                         if latest_ts:
                              data_to_store = data[data['timestamp'] > latest_ts].copy()
                              logger.info(f"Filtré données AV pour {indicator_name} pour ajouter après {latest_ts} ({len(data)} -> {len(data_to_store)} lignes).")
                         else:
                              data_to_store = data.copy()

                         if not data_to_store.empty:
                              # Assurer que la colonne 'value' est là
                              if 'value' in data_to_store.columns:
                                  db_handler.store_dataframe(engine, data_to_store, table_name, if_exists='append')
                              else:
                                  logger.error(f"Colonne 'value' manquante dans les données AV pour indicateur éco {indicator_name}.")
                         else:
                              logger.warning(f"Aucune donnée AV récente à stocker pour indicateur éco {indicator_name}.")
                     else:
                          logger.warning(f"Aucune donnée AV (Eco) retournée ou erreur pour indicateur éco {indicator_name}.")


            # Collecte Commodities (COMMODITIES)
            av_commodities = get_config_list('collector.alpha_vantage', 'commodities')
            if av_commodities:
                 logger.info("--- AV Commodities ---")
                 for commodity_name in av_commodities: # symbol ici est le nom de la commodité AV
                     wait_for_alpha_vantage_rate_limit()
                     table_name = f"av_comm_{commodity_name}".lower()

                     latest_ts = db_handler.get_latest_timestamp(engine, table_name, 'timestamp')
                     # Pour les commodities, la fonction fetch_alpha_vantage_series prend le nom comme 'symbol'
                     data = fetch_alpha_vantage_series(commodity_name, 'COMMODITIES', av_api_key)

                     if data is not None and not data.empty:
                         if latest_ts:
                              data_to_store = data[data['timestamp'] > latest_ts].copy()
                              logger.info(f"Filtré données AV pour {commodity_name} pour ajouter après {latest_ts} ({len(data)} -> {len(data_to_store)} lignes).")
                         else:
                              data_to_store = data.copy()

                         if not data_to_store.empty:
                             if 'value' in data_to_store.columns:
                                  db_handler.store_dataframe(engine, data_to_store, table_name, if_exists='append')
                             else:
                                  logger.error(f"Colonne 'value' manquante dans les données AV pour commodité {commodity_name}.")
                         else:
                              logger.warning(f"Aucune donnée AV récente à stocker pour commodité {commodity_name}.")
                     else:
                          logger.warning(f"Aucune donnée AV (Comm) retournée ou erreur pour commodité {commodity_name}.")


    logger.info("\n" + "="*40)
    logger.info(" FIN DU DATA COLLECTOR ")
    logger.info("="*40 + "\n")


# --- Point d'Entrée Principal ---
if __name__ == "__main__":
    main() 