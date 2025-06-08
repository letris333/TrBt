# labelling.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List
from configparser import ConfigParser, NoSectionError, NoOptionError # Added imports

logger = logging.getLogger(__name__)

def compute_discrete_labels(close_prices: pd.Series, config: ConfigParser) -> pd.Series:
    """
    Calcule les étiquettes discrètes (0: Hold, 1: Buy, 2: Sell) basées sur les rendements futurs.
    Utilise les seuils alpha et beta et la fenêtre forW définis dans la config.
    """
    if close_prices is None or close_prices.empty:
        logger.warning("Série de prix de clôture vide fournie à compute_discrete_labels.")
        return pd.Series(name='discrete_label', dtype=int)

    try:
        # Récupérer les paramètres de configuration avec gestion des erreurs depuis la section [training] ou [labelling] (si conservée)
        # Dans la nouvelle config, on utilise [training] pour les paramètres généraux de labelling historique
        alpha = config.getfloat('training', 'label_alpha_percent', fallback=0.1) / 100.0 # Exemple, seuils plus petits que 1%
        beta = config.getfloat('training', 'label_beta_percent', fallback=0.5) / 100.0   # Exemple
        forW = config.getint('training', 'label_window_periods', fallback=10) # Fenêtre de labellisation, ex: 10 périodes futures

        if not (0 < alpha < beta):
            logger.error(f"Configuration invalide pour alpha ({alpha*100}%) et beta ({beta*100}%). Alpha doit être > 0 et < beta.")
            return pd.Series(name='discrete_label', dtype=int)

        logger.debug(f"Calcul des labels discrets avec alpha={alpha*100:.2f}%, beta={beta*100:.2f}%, forW={forW}")

    except (ValueError, TypeError, NoSectionError, NoOptionError) as e:
        logger.error(f"Erreur de configuration dans labelling config: {e}. Vérifiez les valeurs.")
        return pd.Series(name='discrete_label', dtype=int)

    # Calculer le rendement futur sur la fenêtre forW
    # Utiliser le prix futur divisé par le prix actuel
    future_price_ratio = close_prices.shift(-forW) / close_prices

    # Initialiser les labels à 0 (Hold)
    labels = pd.Series(0, index=close_prices.index, name='discrete_label', dtype=int)

    # Appliquer la logique de labellisation
    # Condition Achat: future_price_ratio est entre (1+alpha) et (1+beta)
    buy_condition = (future_price_ratio > (1 + alpha)) & (future_price_ratio < (1 + beta))
    labels.loc[buy_condition] = 1 # Buy

    # Condition Vente: future_price_ratio est entre (1-beta) et (1-alpha)
    sell_condition = (future_price_ratio < (1 - alpha)) & (future_price_ratio > (1 - beta))
    labels.loc[sell_condition] = 2 # Sell

    # Les labels pour les dernières 'forW' périodes seront NaN par défaut, gérés par dropna.

    label_counts = labels.value_counts().to_dict()
    logger.info(f"Labels discrets calculés - Hold: {label_counts.get(0, 0)}, Buy: {label_counts.get(1, 0)}, Sell: {label_counts.get(2, 0)}")
    return labels


def compute_quantile_targets(close_prices: pd.Series, fq_config: Dict) -> pd.DataFrame:
    """
    Calcule les cibles de quantiles futurs pour le modèle FutureQuant.
    Pour chaque point t, on cherche le prix à t + window_size_out.
    Retourne un DataFrame où chaque ligne t contient les quantiles (calculés sur TOUTES les données historiques)
    du prix à t + window_size_out.

    Alternative plus robuste : Prédire les quantiles du RATIO Price(t+W) / Price(t).
    C'est généralement plus stable que de prédire les prix absolus, surtout si les prix ont une forte tendance.
    Utilisons cette approche.
    """
    if close_prices is None or close_prices.empty:
        logger.warning("Série de prix de clôture vide fournie à compute_quantile_targets.")
        return pd.DataFrame()

    try:
        window_size_out = int(fq_config.get('window_size_out', 5))
        quantiles_list_str = fq_config.get('quantiles', '0.1, 0.5, 0.9')
        quantiles = [float(q.strip()) for q in quantiles_list_str.split(',')]
        if not all(0 <= q <= 1 for q in quantiles):
             logger.error(f"Quantiles invalides: {quantiles}. Doivent être entre 0 et 1.")
             return pd.DataFrame()

        logger.debug(f"Calcul des cibles de quantiles pour FutureQuant (window_size_out={window_size_out}, quantiles={quantiles})")

    except (ValueError, TypeError, NoSectionError, NoOptionError) as e: # Added NoSectionError, NoOptionError
        logger.error(f"Erreur de configuration dans fq_config: {e}. Vérifiez les valeurs.")
        return pd.DataFrame()

    # Calculer le ratio du prix futur par rapport au prix actuel
    future_price_ratio = close_prices.shift(-window_size_out) / close_prices

    # Les cibles sont les quantiles *historiques* de cette série de ratios.
    # Cela suppose que la distribution des rendements futurs est stationnaire.
    # On calcule les quantiles sur l'ensemble des données fournies.
    # C'est une simplification. Idéalement, on entraînerait un modèle pour prédire ces quantiles.
    # L'approche FutureQuant cherche à prédire les quantiles *conditionnels* (conditionnés par la séquence d'entrée).
    # Donc, la *cible* pour une séquence finissant à `t` est le *ratio réel* `Price(t+W) / Price(t)`.
    # La fonction de perte Quantile Loss va ensuite essayer de faire en sorte que les sorties du modèle
    # (les quantiles prédits) "entourent" cette cible réelle de manière appropriée.

    # Donc, la cible est simplement le ratio futur pour chaque point de temps.
    # On créera des colonnes pour chaque quantile prédit, mais la cible d'entraînement est le ratio réel.
    # Le modèle FQ prédira Q10, Q50, Q90 du ratio futur.
    # La cible d'entraînement y_fq sera donc une série/colonne unique: `future_price_ratio`.

    quantile_targets_df = pd.DataFrame(future_price_ratio, index=close_prices.index, columns=['future_price_ratio_target'])

    # Les dernières `window_size_out` périodes auront NaN, gérées par dropna.
    logger.info("Cibles de quantiles (ratios futurs) calculées pour FutureQuant.")
    return quantile_targets_df # Retourne une série (ou DataFrame pour uniformité) 