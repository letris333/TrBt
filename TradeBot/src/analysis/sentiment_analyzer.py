import numpy as np
import pandas as pd
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any
import requests
from configparser import ConfigParser
import os # Import os module
import json # Import json module

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, config_manager: ConfigParser):
        self.config = config_manager
        self._sentiment_pipeline = None
        self._alpaca_api_key: Optional[str] = None
        self._alpaca_secret_key: Optional[str] = None
        self._alpaca_news_base_url: Optional[str] = None
        self._alpaca_news_limit_per_call: int = 50
        self._sentiment_state: Dict[str, Dict[pd.Timestamp, float]] = {} # To store latest sentiment scores
        self._sentiment_state_file: Optional[str] = None
        self._max_sentiment_history: int = 100 # Max points to keep in memory per asset

        self._initialize_sentiment_internal()
        logger.info("SentimentAnalyzer initialized.")

    def _initialize_sentiment_internal(self):
        """
        Initialise le module d'analyse de sentiment.
        Charge le modèle NLP et la configuration de l'API Alpaca.
        """
        logger.info("Initialisation du module d'analyse de sentiment interne...")

        try:
            model_name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            logger.info(f"Modèle de sentiment '{model_name}' chargé avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle de sentiment '{model_name}': {e}", exc_info=True)
            self._sentiment_pipeline = None

        try:
            alpaca_config = self.config.get_section('alpaca')
            if alpaca_config:
                self._alpaca_api_key = alpaca_config.get('api_key')
                self._alpaca_secret_key = alpaca_config.get('secret_key')
                self._alpaca_news_base_url = alpaca_config.get('news_base_url', 'https://data.alpaca.markets/v1beta1')
                self._alpaca_news_limit_per_call = int(alpaca_config.get('news_limit_per_call', 50))

                if not self._alpaca_api_key or not self._alpaca_secret_key:
                    logger.warning("Clés API Alpaca non trouvées dans la configuration. La récupération des actualités Alpaca sera désactivée.")
                else:
                    logger.info("Configuration Alpaca News API chargée.")
            else:
                logger.warning("Section [alpaca] non trouvée dans config.ini. La récupération des actualités Alpaca sera désactivée.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration Alpaca: {e}", exc_info=True)
        
        self._sentiment_state_file = self.config.get('sentiment', 'sentiment_state_file', fallback='data/sentiment_state.json')
        self._max_sentiment_history = self.config.getint('sentiment', 'max_history_points', fallback=100)
        self.load_sentiment_state()


    def fetch_news_for_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Récupère les actualités pour un symbole donné sur une période via l'API Alpaca.
        """
        if not self._alpaca_api_key or not self._alpaca_secret_key or not self._alpaca_news_base_url:
            logger.warning("Configuration Alpaca manquante, impossible de récupérer les actualités.")
            return []

        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        headers = {
            "APCA-API-KEY-ID": self._alpaca_api_key,
            "APCA-API-SECRET-KEY": self._alpaca_secret_key
        }
        params = {
            "symbols": symbol,
            "start": start_str,
            "end": end_str,
            "limit": self._alpaca_news_limit_per_call,
            "include_content": False,
            "sort": "desc"
        }

        all_news_items = []
        page_token = None
        news_endpoint = f"{self._alpaca_news_base_url.rstrip('/')}/news"

        while True:
            if page_token:
                params["page_token"] = page_token
            
            try:
                logger.debug(f"Récupération des actualités Alpaca pour {symbol} avec params: {params}")
                response = requests.get(news_endpoint, headers=headers, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                fetched_news = data.get("news", [])
                if not fetched_news:
                    logger.info(f"Aucune nouvelle actualité Alpaca trouvée pour {symbol} pour cette page.")
                    break

                for news_item in fetched_news:
                    news_timestamp_str = news_item.get('created_at')
                    news_timestamp = pd.to_datetime(news_timestamp_str).replace(tzinfo=timezone.utc) if news_timestamp_str else datetime.now(timezone.utc)
                    
                    all_news_items.append({
                        'timestamp': news_timestamp,
                        'headline': news_item.get('headline', ''),
                        'source': news_item.get('source', 'Alpaca')
                    })
                
                page_token = data.get("next_page_token")
                if not page_token:
                    break

            except requests.exceptions.HTTPError as http_err:
                logger.error(f"Erreur HTTP lors de la récupération des actualités Alpaca: {http_err} - Response: {response.text}")
                break
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Erreur de requête lors de la récupération des actualités Alpaca: {req_err}")
                break
            except Exception as e:
                logger.error(f"Erreur inattendue lors de la récupération des actualités Alpaca: {e}", exc_info=True)
                break
                
        logger.info(f"{len(all_news_items)} actualités récupérées de Alpaca pour {symbol} de {start_date} à {end_date}.")
        return all_news_items

    def analyze_sentiment_of_text(self, text: str) -> float:
        """
        Analyse le sentiment d'un texte donné en utilisant le pipeline Transformers.
        Retourne un score entre -1 (très négatif) et 1 (très positif).
        """
        if not self._sentiment_pipeline:
            logger.warning("Pipeline de sentiment non initialisé. Retourne un score neutre.")
            return 0.0
        if not text or not isinstance(text, str):
            return 0.0

        try:
            result = self._sentiment_pipeline(text)
            
            if result and isinstance(result, list) and len(result) > 0:
                label = result[0]['label'].upper()
                score = result[0]['score']
                
                if label == 'POSITIVE':
                    return score
                elif label == 'NEGATIVE':
                    return -score
                elif label == 'NEUTRAL':
                    return 0.0
                else:
                    logger.warning(f"Label de sentiment inconnu '{label}' reçu du modèle.")
                    return 0.0
            else:
                logger.warning(f"Résultat inattendu de l'analyse de sentiment pour le texte: '{text[:100]}...' Résultat: {result}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de sentiment NLP pour '{text[:100]}...': {e}", exc_info=True)
            return 0.0

    def get_sentiment_data(self, symbol_market: str) -> Dict[str, float]:
        """
        Retourne le dernier score de sentiment pour un actif.
        """
        # Ceci est une version simplifiée pour le trading loop.
        # Pour une analyse complète, il faudrait récupérer les news en temps réel.
        # Pour l'instant, on retourne un score factice ou le dernier score connu.
        
        if symbol_market in self._sentiment_state and self._sentiment_state[symbol_market]:
            # Retourne le dernier score de sentiment stocké
            latest_ts = max(self._sentiment_state[symbol_market].keys())
            return {'score': self._sentiment_state[symbol_market][latest_ts], 'timestamp': latest_ts.isoformat()}
        
        # Si pas de données, générer un score factice pour éviter les erreurs
        # Dans une vraie application, on irait chercher les news ici.
        logger.warning(f"Aucun score de sentiment stocké pour {symbol_market}. Génération d'un score factice.")
        dummy_score = random.uniform(-0.5, 0.5) # Score aléatoire pour le test
        current_ts = pd.Timestamp.now(tz=timezone.utc).floor('min') # Arrondi à la minute
        
        self.add_sentiment_score(symbol_market, current_ts, dummy_score)
        return {'score': dummy_score, 'timestamp': current_ts.isoformat()}

    def add_sentiment_score(self, symbol_market: str, timestamp: pd.Timestamp, score: float) -> None:
        """
        Ajoute un score de sentiment pour un actif à un timestamp donné.
        """
        if symbol_market not in self._sentiment_state:
            self._sentiment_state[symbol_market] = {}
        
        self._sentiment_state[symbol_market][timestamp] = score
        
        # Garder l'historique limité
        timestamps = sorted(self._sentiment_state[symbol_market].keys())
        if len(timestamps) > self._max_sentiment_history:
            for ts_old in timestamps[:-self._max_sentiment_history]:
                del self._sentiment_state[symbol_market][ts_old]
        
        logger.debug(f"Score de sentiment {score:.4f} ajouté pour {symbol_market} à {timestamp}.")

    def load_sentiment_state(self):
        """Charge le dernier état des scores de sentiment depuis le fichier."""
        if self._sentiment_state_file and os.path.exists(self._sentiment_state_file):
            try:
                with open(self._sentiment_state_file, 'r') as f:
                    data = json.load(f)
                    # Convertir les clés de string à Timestamp
                    loaded_state = {}
                    for asset, ts_scores in data.items():
                        loaded_state[asset] = {pd.Timestamp(ts): score for ts, score in ts_scores.items()}
                    self._sentiment_state = loaded_state
                    logger.info(f"Sentiment state chargé depuis {self._sentiment_state_file}.")
            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"Erreur lors du chargement du sentiment state depuis {self._sentiment_state_file}: {e}")
                self._sentiment_state = {}
            except Exception as e:
                logger.error(f"Erreur inattendue lors du chargement du sentiment state: {e}", exc_info=True)
                self._sentiment_state = {}
        else:
            self._sentiment_state = {}
            logger.info("Aucun fichier de sentiment state trouvé. Initialisation à vide.")

    def save_sentiment_state(self):
        """Sauvegarde l'état actuel des scores de sentiment dans le fichier."""
        if self._sentiment_state_file:
            try:
                dirname = os.path.dirname(self._sentiment_state_file)
                if dirname and not os.path.exists(dirname):
                    os.makedirs(dirname)
                
                # Convertir les Timestamps en string pour la sérialisation JSON
                serializable_state = {
                    asset: {str(ts): score for ts, score in ts_scores.items()}
                    for asset, ts_scores in self._sentiment_state.items()
                }

                with open(self._sentiment_state_file, 'w') as f:
                    json.dump(serializable_state, f, indent=4)
                logger.debug(f"Sentiment state sauvegardé dans {self._sentiment_state_file}.")
            except (IOError, TypeError) as e:
                logger.error(f"Erreur lors de la sauvegarde du sentiment state dans {self._sentiment_state_file}: {e}")
            except Exception as e:
                logger.error(f"Erreur inattendue lors de la sauvegarde du sentiment state: {e}", exc_info=True)