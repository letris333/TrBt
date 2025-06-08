import pandas as pd
import logging
from typing import Optional # Ajout pour la cohérence

logger = logging.getLogger(__name__)

# Ajouter ou améliorer la fonction format_symbol utilisée par order_manager.py
def format_symbol(symbol: str, market: str, api = None) -> str:
    """
    Formate un symbole selon les conventions du marché spécifié.
    Cette fonction est utilisée par order_manager.py pour assurer la cohérence des formats.
    
    Args:
        symbol: Le symbole à formater (ex: "BTC/USD" ou "BTC")
        market: Le marché cible ('ccxt' ou 'alpaca_trade')
        api: L'instance API (pour CCXT, peut être nécessaire pour le formatage)
        
    Returns:
        Le symbole correctement formaté pour le marché
    """
    # Déjà formaté correctement?
    if market == 'ccxt' and '/' in symbol:
        return symbol
        
    # CCXT attend généralement BASE/QUOTE
    if market == 'ccxt':
        # Essayer de deviner la paire de trading si le symbole est juste la base
        if '/' not in symbol:
            if symbol.upper() in ['BTC', 'ETH', 'SOL', 'XRP', 'ADA']:
                return f"{symbol.upper()}/USD"  # Paires USD communes
            else:
                return f"{symbol.upper()}/USDT"  # Fallback USDT pour autres crypto
        return symbol
        
    # Alpaca a un format différent pour les actions vs crypto
    elif market == 'alpaca_trade':
        # Pour les cryptomonnaies, Alpaca utilise BTC/USD -> BTCUSD
        if symbol.upper() in ['BTC', 'ETH', 'SOL', 'ADA', 'DOGE', 'AVAX', 'DOT', 'LINK']:
            if '/' in symbol:
                # Retirer le / et gérer USD/USDT
                parts = symbol.upper().split('/')
                if parts[1] == 'USDT':
                    parts[1] = 'USD'
                return f"{parts[0]}{parts[1]}"
            else:
                return f"{symbol.upper()}USD"
        else:
            # Pour les actions, retourner simplement le symbole en majuscules
            return symbol.upper().replace("/", "")
            
    # Fallback: retourner le symbole tel quel
    return symbol


# Améliorer fetch_latest_data pour mieux fonctionner avec db_handler
def fetch_latest_data(api, symbol: str, market: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
    """Récupère les données des N dernières périodes depuis l'API."""
    # Les implémentations peuvent varier selon le marché et l'API
    try:
        if market == 'ccxt':
            # Format OHLCV pour CCXT
            formatted_symbol = format_symbol(symbol, market, api)
            ohlcv = api.fetch_ohlcv(formatted_symbol, timeframe=timeframe, limit=limit)
            
            # Convertir le résultat en DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convertir les timestamps milliseconds -> datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        elif market == 'alpaca_trade':
            # Alpaca a une API différente
            formatted_symbol = format_symbol(symbol, market)
            
            # Convertir timeframe en format Alpaca
            alpaca_timeframe = timeframe
            if timeframe == '1h':
                alpaca_timeframe = '1Hour' # Corrigé: format Alpaca
            elif timeframe == '1d':
                alpaca_timeframe = '1Day'  # Corrigé: format Alpaca
            elif 'min' in timeframe: # Exemple pour les minutes
                 alpaca_timeframe = timeframe.replace('min','Min') # ex: 15min -> 15Min
            # Ajouter d'autres conversions si nécessaire
            
            # Calculer la période
            end = pd.Timestamp.now(tz='UTC')
            
            # Pour le timeframe, déterminer l'unité et le multiplicateur
            if 'h' in timeframe.lower() or 'hour' in timeframe.lower(): # Gérer 'h' et 'Hour'
                unit = 'hours'
                mult_str = timeframe.lower().replace('h', '').replace('our', '')
                mult = int(mult_str) if mult_str.isdigit() else 1
            elif 'd' in timeframe.lower() or 'day' in timeframe.lower(): # Gérer 'd' et 'Day'
                unit = 'days'
                mult_str = timeframe.lower().replace('d', '').replace('ay', '')
                mult = int(mult_str) if mult_str.isdigit() else 1
            elif 'm' in timeframe.lower() or 'min' in timeframe.lower(): # Gérer 'm' et 'Min'
                unit = 'minutes'
                mult_str = timeframe.lower().replace('m', '').replace('in', '')
                mult = int(mult_str) if mult_str.isdigit() else 1
            else: # Fallback
                unit = 'days'
                mult = 1
                
            # Calculer la période de début
            delta = pd.Timedelta(**{unit: mult * limit})
            start = end - delta
            
            # Récupérer les données
            # S'assurer que l'API Alpaca est bien passée et utilisée
            if api is None:
                logger.error("API Alpaca non fournie à fetch_latest_data.")
                return None

            bars = api.get_bars(
                formatted_symbol,
                alpaca_timeframe,
                start=start.isoformat(),
                end=end.isoformat()
            ).df # .df pour obtenir un DataFrame Pandas
            
            # Renommer les colonnes si nécessaire pour correspondre à OHLCV standard
            # L'API Alpaca retourne déjà des noms de colonnes standard (open, high, low, close, volume)
            # L'index est déjà un DatetimeIndex timezone-aware
            
            return bars
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données pour {symbol}@{market}: {e}")
        return None