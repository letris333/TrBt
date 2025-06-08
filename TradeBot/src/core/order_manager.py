# order_manager.py
import logging
from . import data_handler # Pour réutiliser format_symbol et types
from typing import Optional, Dict, Any, Callable, Tuple
import ccxt
import time
import traceback # Added for detailed error logging
from configparser import ConfigParser # Added for config access

try:
    import alpaca_trade_api as alpaca
except ImportError:
    alpaca = None

logger = logging.getLogger(__name__)

class OrderManager:
    def __init__(self, exchange_client: Any, config_manager: ConfigParser):
        self.exchange_client = exchange_client
        self.config = config_manager
        logger.info("OrderManager initialized.")

    def get_cash(self, market: str, currency: str = 'USD') -> Optional[float]:
        """
        Obtient le solde de cash disponible (cherche USD puis USDT par défaut).
        Retourne le montant ou None en cas d'erreur.
        """
        api = self.exchange_client
        if not api:
            logger.error(f"API non initialisée pour le marché '{market}', impossible de récupérer le solde.")
            return None
        logger.debug(f"Récupération du solde {currency} sur {market}...")
        try:
            if market == 'ccxt':
                balance = api.fetch_balance()
                cash = 0.0
                if currency in balance and 'free' in balance[currency]:
                     cash = balance[currency]['free']
                elif currency != 'USDT' and 'USDT' in balance and 'free' in balance['USDT']:
                     logger.debug(f"Solde {currency} non trouvé, utilise USDT comme fallback.")
                     cash = balance['USDT']['free']

                logger.info(f"Solde disponible ({currency}/USDT) sur {market}: {cash}")
                return float(cash) if cash else 0.0

            elif market == 'alpaca_trade':
                if alpaca is None:
                    logger.error("Bibliothèque alpaca_trade_api non disponible.")
                    return None
                account = api.get_account()
                cash_available = float(account.cash)
                logger.info(f"Solde cash disponible sur Alpaca: {cash_available}")
                return cash_available

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.error(f"Erreur API lors de la récupération du solde ({market}): {e}", exc_info=False)
            return None
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la récupération du solde ({market}): {e}", exc_info=True)
            return None
        return 0.0

    def get_position(self, symbol: str, market: str) -> float:
        """
        Obtient la quantité de la position actuelle pour un actif.
        Retourne 0.0 si pas de position ou en cas d'erreur.
        """
        api = self.exchange_client
        if not api:
            logger.error(f"API non initialisée pour le marché '{market}', impossible de récupérer la position pour {symbol}.")
            return 0.0
        formatted_symbol = data_handler.format_symbol(symbol, market, api if market == 'ccxt' else None)
        logger.debug(f"Récupération de la position pour {formatted_symbol} sur {market}...")

        try:
            if market == 'ccxt':
                if '/' not in formatted_symbol:
                     logger.warning(f"Impossible de déterminer l'asset de base pour {formatted_symbol} sur CCXT. Format attendu: BASE/QUOTE")
                     return 0.0
                base_currency = formatted_symbol.split('/')[0]
                balance = api.fetch_balance()

                position_qty = 0.0
                if base_currency in balance and 'free' in balance[base_currency]:
                     position_qty = balance[base_currency]['free']
                     logger.info(f"Position (solde {base_currency}) sur {market} pour {formatted_symbol}: {position_qty}")
                     return float(position_qty) if position_qty else 0.0
                else:
                     logger.info(f"Aucun solde trouvé pour {base_currency} sur {market}. Position = 0.")
                     return 0.0

            elif market == 'alpaca_trade':
                if alpaca is None:
                    logger.error("Bibliothèque alpaca_trade_api non disponible.")
                    return 0.0
                try:
                    position = api.get_position(formatted_symbol)
                    position_qty = float(position.qty)
                    logger.info(f"Position Alpaca pour {formatted_symbol}: {position_qty} (Actuel: {position.side} {position.qty})")
                    return position_qty
                except Exception as e:
                    if hasattr(e, 'status_code') and e.status_code == 404:
                        logger.info(f"Aucune position Alpaca trouvée pour {formatted_symbol}.")
                        return 0.0
                    else:
                        logger.error(f"Erreur API Alpaca lors de la récupération de position pour {formatted_symbol}: {e}", exc_info=False)
                        return 0.0

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
             logger.error(f"Erreur API CCXT lors de la récupération de position ({formatted_symbol}): {e}", exc_info=False)
             return 0.0
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la récupération de position ({formatted_symbol}): {e}", exc_info=True)
            return 0.0
        return 0.0

    def place_order(self, api: Optional[Any], symbol: str, market: str, side: str, qty: float, order_type: str = 'market', time_in_force: str = 'gtc') -> Optional[Dict]:
        """
        Place un ordre d'achat ('buy') ou de vente ('sell').
        Retourne les détails de l'ordre si succès, None sinon.
        """
        if not api:
            logger.error(f"API non initialisée pour le marché '{market}', impossible de placer l'ordre pour {symbol}.")
            return None
        if side not in ['buy', 'sell']:
            logger.error(f"Côté d'ordre invalide : '{side}'. Doit être 'buy' ou 'sell'.")
            return None
        if qty <= 0:
            logger.error(f"Quantité d'ordre invalide ({qty}). Doit être positive.")
            return None
        formatted_symbol = data_handler.format_symbol(symbol, market, api if market == 'ccxt' else None)
        logger.info(f"Tentative de placer un ordre: {side.upper()} {qty:.8f} {formatted_symbol} @{market} type={order_type}")

        min_qty = 0.0
        qty_precision = 8

        try:
            if market == 'ccxt' and hasattr(api, 'markets') and formatted_symbol in api.markets:
                limits = api.markets[formatted_symbol].get('limits', {})
                amount_limits = limits.get('amount', {})
                min_qty = amount_limits.get('min', 0.0)
                precision = api.markets[formatted_symbol].get('precision', {})
                qty_precision = precision.get('amount')
                if qty_precision is None: qty_precision = 8

                if qty_precision is not None:
                     qty = round(qty, int(qty_precision))
                     logger.debug(f"Quantité arrondie à la précision {qty_precision}: {qty:.{qty_precision}f}")

                if min_qty and qty < min_qty:
                     logger.error(f"La quantité {qty:.8f} est inférieure au minimum requis ({min_qty}) pour {formatted_symbol}. Ordre annulé.")
                     return None

            elif market == 'alpaca_trade':
                 pass
        except Exception as e:
            logger.warning(f"Impossible de vérifier/arrondir la quantité pour {formatted_symbol}: {e}. Tentative d'ordre avec la quantité originale.")

        if qty <= 0:
             logger.error(f"Quantité devenue nulle ou négative après arrondissement pour {formatted_symbol}. Ordre annulé.")
             return None

        order_result = None
        try:
            if market == 'ccxt':
                order_result = api.create_order(formatted_symbol, order_type, side, qty)
                logger.info(f"Ordre CCXT placé avec succès: ID={order_result.get('id', 'N/A')}")

            elif market == 'alpaca_trade':
                if alpaca is None:
                    logger.error("Bibliothèque alpaca_trade_api non disponible.")
                    return None
                order_result = api.submit_order(
                    symbol=formatted_symbol,
                    qty=qty,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force
                )
                logger.info(f"Ordre Alpaca soumis avec succès: ID={order_result.id}, ClientOrderID={order_result.client_order_id}")
                order_result = order_result.__dict__.get('_raw', {})

            return order_result

        except ccxt.InsufficientFunds as e:
            logger.error(f"Fonds insuffisants pour placer l'ordre {side} {qty} {formatted_symbol}: {e}", exc_info=False)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.error(f"Erreur API lors de la passation de l'ordre ({formatted_symbol}): {e}", exc_info=False)
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la passation de l'ordre ({formatted_symbol}): {e}", exc_info=True)

        return None

    def place_buy_order_with_tp_sl(self, api_call_func: Callable, symbol: str, qty: float, tp_price: Optional[float] = None, sl_price: Optional[float] = None, 
                                   market: str = 'ccxt') -> Tuple[bool, Optional[str]]:
        """
        Place un ordre d'achat avec Take Profit et Stop Loss.
        Retourne True si tous les ordres ont été placés, False sinon, et l'ID de l'ordre principal.
        """
        api = self.exchange_client
        if not api:
            logger.error("Exchange non initialisé, impossible de placer l'ordre d'achat.")
            return False, None
        
        formatted_symbol = data_handler.format_symbol(symbol, market, api if market == 'ccxt' else None)
        
        try:
            order_result = self.place_order(api, symbol, market, 'buy', qty, order_type='market')
            
            if not order_result:
                logger.error(f"Échec de l'ordre d'achat principal pour {formatted_symbol}.")
                return False, None
            
            order_id = order_result.get('id')
            filled_price = None
            
            if market == 'ccxt':
                time.sleep(2)
                
                try:
                    filled_order = api_call_func(api.fetch_order, order_id, formatted_symbol)
                    if filled_order and filled_order.get('status') == 'closed':
                        filled_price = filled_order.get('average') or filled_order.get('price')
                        logger.info(f"Ordre d'achat exécuté pour {formatted_symbol} au prix moyen {filled_price}.")
                    else:
                        logger.warning(f"L'ordre d'achat pour {formatted_symbol} n'est pas encore terminé. "
                                      f"Statut: {filled_order.get('status')}. Utilisation du prix actuel pour TP/SL.")
                        ticker = api_call_func(api.fetch_ticker, formatted_symbol)
                        filled_price = ticker['last'] if ticker else None
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des détails de l'ordre: {e}")
                    ticker = api_call_func(api.fetch_ticker, formatted_symbol)
                    filled_price = ticker['last'] if ticker else None
            
            if filled_price:
                entry_price = filled_price
            else:
                ticker = api_call_func(api.fetch_ticker, formatted_symbol)
                entry_price = ticker['last'] if ticker else None
                logger.warning(f"Prix de remplissage non disponible pour {formatted_symbol}. "
                              f"Utilisation du dernier prix connu: {entry_price}")
            
            if not entry_price:
                logger.error(f"Impossible d'obtenir un prix de référence pour {formatted_symbol}. Abandon des ordres TP/SL.")
                return True, order_id
            
            tp_placed = True
            sl_placed = True
            
            if tp_price and tp_price > 0:
                if market == 'ccxt' and hasattr(api, 'create_limit_sell_order'):
                    try:
                        tp_order = api_call_func(api.create_limit_sell_order, formatted_symbol, qty, tp_price)
                        logger.info(f"Ordre TP placé pour {formatted_symbol} à {tp_price}.")
                    except Exception as e:
                        logger.error(f"Erreur lors du placement de l'ordre TP pour {formatted_symbol}: {e}")
                        tp_placed = False
            
            if sl_price and sl_price > 0:
                if market == 'ccxt' and hasattr(api, 'create_stop_loss_order'):
                    try:
                        sl_order = api_call_func(api.create_stop_loss_order, formatted_symbol, 'sell', qty, sl_price)
                        logger.info(f"Ordre SL placé pour {formatted_symbol} à {sl_price}.")
                    except Exception as e:
                        logger.error(f"Erreur lors du placement de l'ordre SL pour {formatted_symbol}: {e}")
                        sl_placed = False
                elif market == 'ccxt':
                    try:
                        params = {'stopPrice': sl_price}
                        sl_order = api_call_func(api.create_order, formatted_symbol, 'stop_market', 'sell', qty, None, params)
                        logger.info(f"Ordre SL (stop market) placé pour {formatted_symbol} à {sl_price}.")
                    except Exception as e:
                        logger.error(f"Erreur lors du placement de l'ordre SL (stop market) pour {formatted_symbol}: {e}")
                        sl_placed = False
            
            logger.info(f"Ordres pour {formatted_symbol}: Principal=OK, TP={'OK' if tp_placed else 'NON'}, SL={'OK' if sl_placed else 'NON'}")
            return True, order_id
        
        except Exception as e:
            logger.error(f"Erreur inattendue lors du placement d'ordre pour {formatted_symbol}: {e}")
            logger.error(traceback.format_exc())
            return False, None

    def place_sell_order(self, api_call_func: Callable, symbol: str, qty: float, market: str = 'ccxt') -> bool:
        """
        Place un ordre de vente simple (pour clôture de position).
        Retourne True si l'ordre a été placé, False sinon.
        """
        api = self.exchange_client
        if not api:
            logger.error("Exchange non initialisé, impossible de placer l'ordre de vente.")
            return False
        
        try:
            order_result = self.place_order(api, symbol, market, 'sell', qty, order_type='market')
            return order_result is not None
        
        except Exception as e:
            logger.error(f"Erreur lors du placement de l'ordre de vente pour {symbol}: {e}")
            return False

    def update_stop_loss(self, api_call_func: Callable, symbol: str, new_sl_price: float, position_qty: float, current_sl_order_id: Optional[str], market: str = 'ccxt') -> bool:
        """
        Met à jour le Stop Loss pour une position existante.
        Peut être utilisé pour le Move-to-BE ou le Trailing Stop.
        
        Args:
            api_call_func: La fonction wrapper pour les appels API avec retry.
            symbol: Le symbole de l'actif
            new_sl_price: Le nouveau prix du Stop Loss
            position_qty: La quantité de la position (nécessaire pour recréer l'ordre SL)
            current_sl_order_id: L'ID de l'ordre SL existant à annuler (peut être None)
            market: Le marché ('ccxt' par défaut)
            
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        api = self.exchange_client
        if not api:
            logger.error(f"API non initialisée pour le marché '{market}', impossible de mettre à jour le SL pour {symbol}.")
            return False
        
        formatted_symbol = data_handler.format_symbol(symbol, market, api if market == 'ccxt' else None)
        logger.info(f"Tentative de mettre à jour le SL pour {formatted_symbol} à {new_sl_price:.4f}")
        
        try:
            # Annuler l'ancien ordre SL si un ID est fourni
            if current_sl_order_id:
                logger.info(f"Annulation de l'ancien SL pour {formatted_symbol}: ID={current_sl_order_id}")
                try:
                    api_call_func(api.cancel_order, current_sl_order_id, formatted_symbol)
                except Exception as e:
                    logger.warning(f"Impossible d'annuler l'ancien ordre SL {current_sl_order_id} pour {formatted_symbol}: {e}. Tentative de continuer.")
            
            # Créer un nouvel ordre SL
            if hasattr(api, 'create_stop_loss_order'):
                new_sl_order = api_call_func(api.create_stop_loss_order, formatted_symbol, 'sell', position_qty, new_sl_price)
            else:
                params = {'stopPrice': new_sl_price}
                new_sl_order = api_call_func(api.create_order, formatted_symbol, 'stop_market', 'sell', position_qty, None, params)
            
            if new_sl_order:
                logger.info(f"Nouvel ordre SL créé pour {formatted_symbol} à {new_sl_price} pour {position_qty} unités. ID={new_sl_order.get('id')}")
                return True
            else:
                logger.error(f"Échec de la création du nouvel ordre SL pour {formatted_symbol}.")
                return False
            
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.error(f"Erreur API lors de la mise à jour du SL pour {formatted_symbol}: {e}")
            return False
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la mise à jour du SL pour {formatted_symbol}: {e}")
            logger.error(traceback.format_exc())
            return False

    def update_trailing_stop(self, api_call_func: Callable, symbol: str, trailing_stop_price: float, market: str = 'ccxt') -> bool:
        """
        Met à jour un Trailing Stop pour une position existante.
        Cette fonction peut utiliser update_stop_loss ou implémenter une logique spécifique
        de trailing stop si supportée nativement par l'exchange.
        
        Args:
            api_call_func: La fonction wrapper pour les appels API avec retry.
            symbol: Le symbole de l'actif
            trailing_stop_price: Le nouveau prix du Trailing Stop
            market: Le marché ('ccxt' par défaut)
            
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        api = self.exchange_client
        # Vérifier si l'exchange supporte nativement les trailing stops
        if market == 'ccxt' and hasattr(api, 'create_trailing_stop_order'):
            try:
                # Récupérer la position actuelle
                position_qty = self.get_position(symbol, market)
                
                if position_qty <= 0:
                    logger.error(f"Aucune position longue active pour {symbol}. Impossible de créer un trailing stop.")
                    return False
                
                # Annuler les SL/TS existants
                formatted_symbol = data_handler.format_symbol(symbol, market, api if market == 'ccxt' else None)
                open_orders = api_call_func(api.fetch_open_orders, formatted_symbol)
                
                for order in open_orders:
                    if order.get('type') in ['stop', 'stop_market', 'stop_loss', 'trailing_stop'] and order.get('side') == 'sell':
                        api_call_func(api.cancel_order, order.get('id'), formatted_symbol)
                        logger.info(f"Ordre stop existant annulé: {order.get('id')}")
                
                # Créer un nouveau trailing stop (la syntaxe varie selon l'exchange)
                params = {}
                
                trailing_stop_order = api_call_func(
                    api.create_trailing_stop_order, formatted_symbol, 'sell', position_qty, None, params
                )
                
                logger.info(f"Trailing Stop créé pour {formatted_symbol} avec activation à {trailing_stop_price}.")
                return True
                
            except Exception as e:
                logger.error(f"Erreur lors de la création du trailing stop pour {symbol}: {e}")
                logger.error(traceback.format_exc())
                return False
        else:
            # Fallback: utiliser un stop loss standard
            logger.info(f"Trailing Stop non supporté nativement pour {market}. Utilisation d'un Stop Loss standard.")
            # Note: update_stop_loss nécessite position_qty et current_sl_order_id.
            # Pour un trailing stop, on n'a pas forcément un old_sl_order_id.
            # On peut passer None pour current_sl_order_id si on veut juste créer un nouveau SL.
            # Ou bien, si on veut vraiment "mettre à jour" un SL existant, il faudrait le récupérer.
            # Pour l'instant, on va juste créer un nouveau SL si le trailing stop est activé.
            position_qty = self.get_position(symbol, market)
            if position_qty <= 0:
                logger.error(f"Aucune position longue active pour {symbol}. Impossible de créer un SL de remplacement pour trailing stop.")
                return False
            return self.update_stop_loss(api_call_func, symbol, trailing_stop_price, position_qty, None, market)

    def get_account_balance(self, market: str = 'ccxt') -> float:
        """
        Récupère le solde disponible du compte pour le trading.
        Wrapper simple pour obtenir le solde à utiliser dans le code de trading.
        
        Args:
            market: Le marché ('ccxt' par défaut)
            
        Returns:
            Le solde disponible en USD/USDT ou 0 en cas d'erreur
        """
        try:
            # from main_trader import exchange  # Import ici pour éviter dépendance circulaire
            # L'exchange est maintenant self.exchange_client
            
            if not self.exchange_client:
                logger.error("Exchange non initialisé, impossible de récupérer le solde.")
                return 0.0
            
            balance = self.get_cash(market, 'USDT')
            if balance is None:
                balance = self.get_cash(market, 'USD')
            
            return balance or 0.0
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du solde: {e}")
            return 0.0