import sqlalchemy
from sqlalchemy import create_engine, text, exc
import pandas as pd
import logging
from configparser import ConfigParser
from typing import Optional, Dict
import time
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Dictionnaire pour stocker les engines de DB (un par configuration si nécessaire)
_db_engines: Dict[str, sqlalchemy.engine.Engine] = {}

def get_db_engine(config: ConfigParser) -> Optional[sqlalchemy.engine.Engine]:
    """
    Crée ou retourne un engine SQLAlchemy pour la base de données.
    Retourne None en cas d'erreur.
    """
    db_type = config.get('database', 'db_type', fallback='postgresql')
    db_key = f"{db_type}" # Clé simple pour le cache, peut être plus complexe si configs DB multiples

    if db_key in _db_engines:
        try:
            # Tenter une connexion pour vérifier si l'engine est toujours valide
            with _db_engines[db_key].connect() as connection:
                 connection.execute(text("SELECT 1"))
            logger.debug(f"Engine DB ({db_type}) trouvé dans le cache et connexion vérifiée.")
            return _db_engines[db_key]
        except exc.SQLAlchemyError as e:
            logger.warning(f"Engine DB ({db_type}) dans le cache invalide ({e}). Tentative de recréer.")
            del _db_engines[db_key] # Supprimer l'engine invalide


    db_connection_string = None
    try:
        if db_type == 'postgresql':
            db_host = config.get('database', 'db_host', fallback='localhost')
            db_port = config.get('database', 'db_port', fallback='5432')
            db_name = config.get('database', 'db_name')
            db_user = config.get('database', 'db_user')
            db_password = config.get('database', 'db_password')
            db_connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        elif db_type == 'mysql':
             db_host = config.get('database', 'db_host', fallback='localhost')
             db_port = config.get('database', 'db_port', fallback='3306')
             db_name = config.get('database', 'db_name')
             db_user = config.get('database', 'db_user')
             db_password = config.get('database', 'db_password')
             db_connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        elif db_type == 'sqlite':
             db_path = config.get('database', 'db_path', fallback='trading_data.db')
             # S'assurer que le répertoire existe
             dirname = os.path.dirname(db_path)
             if dirname and not os.path.exists(dirname):
                  os.makedirs(dirname)
             db_connection_string = f"sqlite:///{db_path}"
        else:
            logger.error(f"Type de base de données non supporté dans la config: {db_type}")
            return None

        logger.info(f"Création de l'engine DB pour {db_type}...")
        # pool_pre_ping=True pour vérifier la connexion avant usage
        engine = create_engine(db_connection_string, pool_pre_ping=True)

        # Tester la connexion
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Connexion à la base de données réussie.")
        _db_engines[db_key] = engine # Mettre en cache l'engine
        return engine

    except exc.SQLAlchemyError as e:
        logger.critical(f"Échec de la connexion ou de la création de l'engine DB: {e}")
        return None
    except Exception as e:
        logger.critical(f"Erreur inattendue lors de l'initialisation de la DB: {e}", exc_info=True)
        return None

def store_dataframe(engine: sqlalchemy.engine.Engine, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> bool:
    """
    Stocke un DataFrame dans une table de la base de données.
    Gère les retries pour les erreurs de connexion ou de DB.
    """
    if df is None or df.empty:
        logger.warning(f"Aucune donnée à stocker pour la table '{table_name}'.")
        return False

    # Nettoyer le nom de la table pour assurer la compatibilité (SQL injection basique, caractères spéciaux)
    safe_table_name = ''.join(c for c in table_name if c.isalnum() or c in ('_',))
    if safe_table_name != table_name:
        logger.warning(f"Nom de table '{table_name}' nettoyé en '{safe_table_name}'.")
        table_name = safe_table_name

    # S'assurer que les noms de colonnes sont compatibles avec SQL
    original_columns = list(df.columns)
    df.columns = [col.replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '').replace(')', '').replace('$', '_').lower() for col in df.columns]
    if original_columns != list(df.columns):
        logger.warning(f"Noms de colonnes pour '{table_name}' modifiés pour compatibilité SQL.")


    max_retries = 3
    retry_delay = 5 # seconds

    for attempt in range(max_retries):
        try:
            logger.info(f"Tentative {attempt+1}/{max_retries}: Stockage de {len(df)} lignes dans '{table_name}'...")
            # Utiliser index=True si l'index est pertinent (ex: timestamp) et si vous voulez qu'il devienne une colonne
            # Si l'index est déjà inclus dans le DF, utiliser index=False
            # Assumons que le timestamp est déjà une colonne nommée 'timestamp' ou 'date'
            df.to_sql(table_name, engine, if_exists=if_exists, index=False)
            logger.info(f"Stockage réussi pour la table '{table_name}'.")
            return True # Succès

        except exc.SQLAlchemyError as e:
            logger.error(f"Tentative {attempt+1}/{max_retries} - Erreur DB lors du stockage dans '{table_name}': {e}")
            if attempt + 1 < max_retries:
                logger.info(f"Attente de {retry_delay}s avant de réessayer...")
                time.sleep(retry_delay)
            else:
                logger.critical(f"Échec final du stockage dans '{table_name}' après {max_retries} tentatives.")
                return False # Échec après retries
        except Exception as e:
            logger.error(f"Tentative {attempt+1}/{max_retries} - Erreur inattendue lors du stockage dans '{table_name}': {e}", exc_info=True)
            if attempt + 1 < max_retries:
                logger.info(f"Attente de {retry_delay}s avant de réessayer...")
                time.sleep(retry_delay)
            else:
                logger.critical(f"Échec final inattendu du stockage dans '{table_name}' après {max_retries} tentatives.")
                return False

def get_latest_timestamp(engine: sqlalchemy.engine.Engine, table_name: str, timestamp_column: str = 'timestamp') -> Optional[datetime]:
    """
    Récupère le timestamp le plus récent dans une table.
    Retourne None si la table n'existe pas, est vide ou en cas d'erreur.
    """
    safe_table_name = ''.join(c for c in table_name if c.isalnum() or c in ('_',))
    safe_timestamp_column = ''.join(c for c in timestamp_column if c.isalnum() or c in ('_',))

    if not safe_table_name or not safe_timestamp_column:
         logger.error(f"Nom de table ('{table_name}') ou de colonne timestamp ('{timestamp_column}') invalide après nettoyage.")
         return None

    query = text(f"SELECT MAX({safe_timestamp_column}) FROM {safe_table_name}")

    max_retries = 3
    retry_delay = 2 # seconds

    for attempt in range(max_retries):
        try:
            with engine.connect() as connection:
                result = connection.execute(query).scalar()
                if result is not None:
                     logger.debug(f"Dernier timestamp dans '{table_name}': {result}")
                     # Convertir en datetime si ce n'est pas déjà le cas
                     if isinstance(result, (int, float)): # Potentiel timestamp unix
                          try: return datetime.fromtimestamp(result, tz=timezone.utc) # Assumer UTC
                          except (ValueError, TypeError): pass # Échec de conversion, essayer autre chose
                     if isinstance(result, str): # Potentiel string date
                          try: return pd.to_datetime(result, utc=True) # Tenter conversion
                          except (ValueError, TypeError): pass
                     if isinstance(result, datetime):
                          # Assurer que c'est timezone-aware, idéalement UTC
                          if result.tzinfo is None:
                               logger.warning(f"Timestamp sans timezone pour '{table_name}'. Assumé UTC.")
                               return result.replace(tzinfo=timezone.utc)
                          return result.astimezone(timezone.utc) # Convertir en UTC
                     # Si c'est déjà un type date/datetime de la DB, SQLAlchemy le gère bien
                     return result # Espérons que c'est un datetime timezone-aware

                else:
                    logger.info(f"La table '{table_name}' est vide ou n'existe pas.")
                    return None

        except exc.ProgrammingError as e:
            # Table n'existe pas
            if "relation" in str(e).lower() and ("does not exist" in str(e).lower() or "doesn't exist" in str(e).lower()):
                 logger.info(f"La table '{table_name}' n'existe pas encore.")
                 return None
            else:
                 logger.error(f"Tentative {attempt+1}/{max_retries} - Erreur de programmation DB pour '{table_name}': {e}")
        except exc.SQLAlchemyError as e:
            logger.error(f"Tentative {attempt+1}/{max_retries} - Erreur DB lors de la récupération max timestamp pour '{table_name}': {e}")
        except Exception as e:
            logger.error(f"Tentative {attempt+1}/{max_retries} - Erreur inattendue lors de la récupération max timestamp ({table_name}): {e}", exc_info=True)

        if attempt + 1 < max_retries:
             logger.info(f"Attente de {retry_delay}s avant de réessayer...")
             time.sleep(retry_delay)

    logger.critical(f"Échec final de la récupération max timestamp pour '{table_name}' après {max_retries} tentatives.")
    return None # Échec après retries 