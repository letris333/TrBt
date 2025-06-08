# position_manager.py
import logging
import json
import os
from typing import Dict, Optional, List, Any, Callable
from configparser import ConfigParser

logger = logging.getLogger(__name__)

# Dictionnaire pour stocker les positions ouvertes en mémoire
# Format: {'SYMBOL@market': {'qty': float, 'entry_price': float, 'side': 'long'/'short', 'tp': float, 'sl': float}}
_open_positions: Dict[str, Dict] = {}
_positions_state_file: Optional[str] = None

def initialize_position_state(config: ConfigParser):
    """Initialise le chemin du fichier de state et charge le state si existant."""
    global _positions_state_file
    _positions_state_file = config.get('strategy_hybrid', 'positions_state_file', fallback='open_positions_state.json')
    load_positions_state()
    logger.info(f"Open positions state initialized. File: {_positions_state_file}")

def load_positions_state():
    """Charge le dernier état des positions ouvertes depuis le fichier."""
    global _open_positions
    if _positions_state_file and os.path.exists(_positions_state_file):
        try:
            with open(_positions_state_file, 'r') as f:
                _open_positions = json.load(f)
                logger.info(f"Open positions state chargé depuis {_positions_state_file}. Positions: {len(_open_positions)}")
                for asset, pos_data in _open_positions.items():
                     if 'qty' in pos_data: pos_data['qty'] = float(pos_data['qty'])
                     if 'entry_price' in pos_data: pos_data['entry_price'] = float(pos_data['entry_price'])
                     if 'tp' in pos_data and pos_data['tp'] is not None: pos_data['tp'] = float(pos_data['tp'])
                     if 'sl' in pos_data and pos_data['sl'] is not None: pos_data['sl'] = float(pos_data['sl'])

        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Erreur lors du chargement des positions state depuis {_positions_state_file}: {e}")
            _open_positions = {}
        except Exception as e:
            logger.error(f"Erreur inattendue lors du chargement des positions state: {e}", exc_info=True)
            _open_positions = {}
    else:
        _open_positions = {}
        logger.info("Aucun fichier de positions state trouvé. Initialisation à vide.")


class PositionManager:
    def __init__(self, exchange_client: Any, config_manager: ConfigParser):
        self.exchange_client = exchange_client
        self.config = config_manager
        initialize_position_state(self.config)
        logger.info("PositionManager initialized.")

    def save_positions_state(self):
        """Sauvegarde l'état actuel des positions ouvertes dans le fichier."""
        if _positions_state_file:
            try:
                dirname = os.path.dirname(_positions_state_file)
                if dirname and not os.path.exists(dirname):
                     os.makedirs(dirname)
                with open(_positions_state_file, 'w') as f:
                    json.dump(_open_positions, f, indent=4)
                logger.debug(f"Open positions state sauvegardé dans {_positions_state_file}.")
            except (IOError, TypeError) as e:
                logger.error(f"Erreur lors de la sauvegarde des positions state dans {_positions_state_file}: {e}")
            except Exception as e:
                logger.error(f"Erreur inattendue lors de la sauvegarde des positions state: {e}", exc_info=True)

    def add_position(self, symbol_market: str, entry_price: float, qty: float, side: str, 
                     tp: Optional[float], sl: Optional[float], setup_quality: float, 
                     risk_multiplier: float, order_id: Optional[str] = None, 
                     exit_strategy_params: Optional[Dict] = None):
        """Ajoute une nouvelle position ouverte à la liste."""
        if symbol_market in _open_positions:
            logger.warning(f"Tentative d'ajouter une position pour {symbol_market} qui existe déjà. Mise à jour.")
        
        if qty > 0 and entry_price > 0 and side in ['long', 'short']:
            _open_positions[symbol_market] = {
                'qty': qty,
                'entry_price': entry_price,
                'side': side,
                'tp': tp,
                'sl': sl,
                'setup_quality': setup_quality,
                'risk_multiplier': risk_multiplier,
                'order_id': order_id,
                'highest_price_seen': entry_price,
                'exit_strategy': exit_strategy_params,
                'status': 'open'
            }
            logger.info(f"Position pour {symbol_market} ajoutée: Qty={qty:.8f}, Entry={entry_price:.4f}, Side={side}, TP={tp}, SL={sl}")
        else:
             logger.error(f"Paramètres invalides pour ajouter une position pour {symbol_market}. Qty: {qty}, Entry: {entry_price}, Side: {side}")

    def remove_position(self, symbol_market: str, closure_price: Optional[float] = None):
        """Retire une position ouverte de la liste et log le PnL."""
        if symbol_market in _open_positions:
            pos_data = _open_positions[symbol_market]
            entry_price = pos_data['entry_price']
            qty = pos_data['qty']
            side = pos_data['side']
            
            pnl_abs = 0.0
            pnl_pct = 0.0
            if closure_price is not None and closure_price > 0:
                if side == 'long':
                    pnl_abs = (closure_price - entry_price) * qty
                    pnl_pct = (closure_price - entry_price) / entry_price
                elif side == 'short':
                    pnl_abs = (entry_price - closure_price) * qty
                    pnl_pct = (entry_price - closure_price) / entry_price
                logger.info(f"Position pour {symbol_market} clôturée à {closure_price:.4f}. PnL: {pnl_abs:.2f} ({pnl_pct:.2%}).")
            else:
                logger.info(f"Position pour {symbol_market} retirée (prix de clôture non spécifié).")

            del _open_positions[symbol_market]
        else:
            logger.warning(f"Tentative de retirer une position pour {symbol_market} qui n'existe pas.")

    def get_position_state(self, symbol_market: str) -> Optional[Dict]:
        """Retourne les détails d'une position ouverte ou None si n'existe pas."""
        return _open_positions.get(symbol_market)

    def get_all_open_positions(self) -> Dict[str, Dict]:
        """Retourne toutes les positions ouvertes trackées par le manager."""
        return _open_positions

    def get_active_positions_count(self) -> int:
        """Retourne le nombre de positions actives (non en attente de clôture)."""
        return len([pos for pos in _open_positions.values() if pos.get('status') == 'open'])

    def update_position_sl(self, symbol_market: str, new_sl: float):
        """Met à jour le Stop Loss d'une position existante."""
        if symbol_market in _open_positions:
            _open_positions[symbol_market]['sl'] = new_sl
            logger.debug(f"SL pour {symbol_market} mis à jour à {new_sl:.4f}.")
        else:
            logger.warning(f"Impossible de mettre à jour le SL pour {symbol_market}: position non trouvée.")

    def update_position_highest_price(self, symbol_market: str, current_price: float):
        """Met à jour le prix le plus haut atteint par une position longue pour le trailing stop."""
        if symbol_market in _open_positions and _open_positions[symbol_market]['side'] == 'long':
            current_highest = _open_positions[symbol_market].get('highest_price_seen', 0.0)
            if current_price > current_highest:
                _open_positions[symbol_market]['highest_price_seen'] = current_price
                logger.debug(f"Highest price seen for {symbol_market} updated to {current_price:.4f}.")

    def update_position_field(self, symbol_market: str, field: str, value: Any):
        """Met à jour un champ spécifique d'une position."""
        if symbol_market in _open_positions:
            _open_positions[symbol_market][field] = value
            logger.debug(f"Champ '{field}' de la position {symbol_market} mis à jour à {value}.")
        else:
            logger.warning(f"Impossible de mettre à jour le champ '{field}' pour {symbol_market}: position non trouvée.")

    def mark_position_closure_pending(self, symbol_market: str):
        """Marque une position comme en attente de clôture."""
        if symbol_market in _open_positions:
            _open_positions[symbol_market]['status'] = 'closure_pending'
            logger.info(f"Position {symbol_market} marquée comme 'closure_pending'.")
        else:
            logger.warning(f"Impossible de marquer {symbol_market} comme 'closure_pending': position non trouvée.")

    def get_daily_pnl(self) -> float:
        """
        Calcule le PnL quotidien agrégé de toutes les positions ouvertes.
        Ceci est une simplification et ne représente pas un PnL réalisé.
        """
        return 0.0

    def fetch_current_positions_from_exchange(self, api_call_func: Callable):
        """
        Récupère les positions ouvertes réelles de l'exchange et les synchronise.
        Ceci est une implémentation simplifiée.
        """
        api = self.exchange_client
        if not api:
            logger.warning("Exchange client non initialisé, impossible de récupérer les positions de l'exchange.")
            return

        logger.info("Synchronisation des positions avec l'exchange...")
        try:
            logger.warning("La synchronisation des positions réelles de l'exchange n'est pas entièrement implémentée pour tous les types de marchés/exchanges.")
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des positions de l'exchange: {e}", exc_info=True)