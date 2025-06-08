import logging
from typing import Dict, Tuple, Optional, List, Any
from configparser import ConfigParser, NoSectionError, NoOptionError
import numpy as np
from src import indicators
from src.core import position_manager

logger = logging.getLogger(__name__)

class StrategyHybrid:
    def __init__(self, config_manager: ConfigParser, indicators_manager_instance: indicators.IndicatorsManager):
        self.config = config_manager
        self.indicators_manager = indicators_manager_instance
        logger.info("StrategyHybrid initialized.")

    def calculate_fib_tp_sl_prices(self, entry_price: float, is_long: bool, sl_distance_percent: float, tp_sl_ratio: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Calcule les prix du Stop-Loss et Take-Profit basés sur un pourcentage
        de distance SL par rapport à l'entrée et un ratio TP/SL Fibonacci.
        Retourne (tp_price, sl_price). None si calcul impossible.
        """
        if entry_price is None or entry_price <= 0 or sl_distance_percent is None or sl_distance_percent <= 0 or tp_sl_ratio is None or tp_sl_ratio <= 0:
            logger.error("Paramètres invalides pour le calcul des niveaux Fibonacci.")
            return (None, None)

        sl_distance_abs = entry_price * sl_distance_percent
        tp_distance_abs = sl_distance_abs * tp_sl_ratio

        if is_long:
            sl_price = entry_price - sl_distance_abs
            tp_price = entry_price + tp_distance_abs
            if sl_price <= 0:
                logger.warning(f"Calcul SL Long <= 0 ({sl_price:.4f}). Entry: {entry_price:.4f}, SL Dist: {sl_distance_abs:.4f}. SL price set to a small positive value.")
                sl_price = entry_price * 0.01
        else:
            sl_price = entry_price + sl_distance_abs
            tp_price = entry_price - tp_distance_abs
            if tp_price < 0:
                 logger.warning(f"Calcul TP Short < 0 ({tp_price:.4f}). Entry: {entry_price:.4f}, TP Dist: {tp_distance_abs:.4f}. TP price set to 0.")
                 tp_price = 0.0

        logger.debug(f"Fib TP/SL: Entry={entry_price:.4f}, SL_dist={sl_distance_abs:.4f}, TP_dist={tp_distance_abs:.4f} => TP={tp_price:.4f}, SL={sl_price:.4f} (Long: {is_long})")
        return (tp_price, sl_price)

    def calculate_atr_tp_sl_prices(self, entry_price: float, is_long: bool, atr_value: float, atr_sl_multiplier: float, tp_sl_ratio: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Calcule les prix du Stop-Loss et Take-Profit basés sur l'ATR.
        Retourne (tp_price, sl_price). None si calcul impossible.
        """
        if not all([entry_price, atr_value, atr_sl_multiplier, tp_sl_ratio]) or entry_price <= 0 or atr_value <= 0 or atr_sl_multiplier <= 0 or tp_sl_ratio <= 0:
            logger.error(f"Paramètres invalides pour le calcul des niveaux TP/SL basés sur ATR. Entry: {entry_price}, ATR: {atr_value}, SL Mult: {atr_sl_multiplier}, TP Ratio: {tp_sl_ratio}")
            return (None, None)

        sl_distance_atr = atr_value * atr_sl_multiplier
        tp_distance_abs = sl_distance_atr * tp_sl_ratio

        if is_long:
            sl_price = entry_price - sl_distance_atr
            tp_price = entry_price + tp_distance_abs
            if sl_price <= 0:
                logger.warning(f"Calcul ATR SL Long <= 0 ({sl_price:.4f}). Entry: {entry_price:.4f}, ATR: {atr_value:.4f}, SL Multiplier: {atr_sl_multiplier:.2f}. SL price set to a small positive value.")
                sl_price = entry_price * 0.001
        else:
            sl_price = entry_price + sl_distance_atr
            tp_price = entry_price - tp_distance_abs
            if tp_price < 0:
                 logger.warning(f"Calcul ATR TP Short < 0 ({tp_price:.4f}). Entry: {entry_price:.4f}, ATR: {atr_value:.4f}, SL Multiplier: {atr_sl_multiplier:.2f}. TP price set to 0.")
                 tp_price = 0.0

        logger.debug(f"ATR TP/SL: Entry={entry_price:.4f}, ATR={atr_value:.4f}, SL_Multiplier={atr_sl_multiplier:.2f}, TP_Ratio={tp_sl_ratio:.2f} => TP={tp_price:.4f}, SL={sl_price:.4f} (Long: {is_long})")
        return (tp_price, sl_price)

    def analyze_ms_for_trade_signal(self, ms_features: Dict, current_price: float) -> Dict:
        """
        Analyse les features Market Structure pour obtenir des signaux de trading.
        """
        ms_signals = {
            'ms_buy_signal': False,
            'ms_sell_signal': False,
            'ms_neutral': True,
            'ms_reference_level': None,
            'ms_confluence_score': 0.0
        }
        
        if not ms_features:
            return ms_signals
        
        poc_price = ms_features.get('ms_poc_price')
        va_high = ms_features.get('ms_va_high_price')
        va_low = ms_features.get('ms_va_low_price')
        hvn_prices = ms_features.get('ms_hvn_prices', [])
        lvn_prices = ms_features.get('ms_lvn_prices', [])
        
        rejection_va_high = ms_features.get('ms_recent_rejection_va_high', False)
        rejection_va_low = ms_features.get('ms_recent_rejection_va_low', False)
        
        price_vs_poc = ms_features.get('ms_price_vs_poc')
        price_vs_va_high = ms_features.get('ms_price_vs_va_high')
        price_vs_va_low = ms_features.get('ms_price_vs_va_low')
        
        buy_confluence_score = 0.0
        sell_confluence_score = 0.0
        
        if va_low is not None and 0 < current_price - va_low < 0.01 * current_price:
            buy_confluence_score += 2.0
            ms_signals['ms_reference_level'] = va_low
        
        if hvn_prices:
            hvn_below = [p for p in hvn_prices if p < current_price and current_price - p < 0.015 * current_price]
            if hvn_below:
                closest_hvn = max(hvn_below)
                buy_confluence_score += 1.5
                if ms_signals['ms_reference_level'] is None:
                    ms_signals['ms_reference_level'] = closest_hvn
        
        if rejection_va_low:
            buy_confluence_score += 1.0
        
        if va_high is not None and 0 < va_high - current_price < 0.01 * current_price:
            sell_confluence_score += 2.0
            ms_signals['ms_reference_level'] = va_high
        
        if hvn_prices:
            hvn_above = [p for p in hvn_prices if p > current_price and p - current_price < 0.015 * current_price]
            if hvn_above:
                closest_hvn_above = min(hvn_above)
                sell_confluence_score += 1.5
                if ms_signals['ms_reference_level'] is None:
                    ms_signals['ms_reference_level'] = closest_hvn_above
        
        if rejection_va_high:
            sell_confluence_score += 1.0
        
        ms_signals['ms_buy_signal'] = buy_confluence_score >= 2.5
        ms_signals['ms_sell_signal'] = sell_confluence_score >= 2.5
        ms_signals['ms_neutral'] = not (ms_signals['ms_buy_signal'] or ms_signals['ms_sell_signal'])
        
        ms_signals['ms_confluence_score'] = max(buy_confluence_score, sell_confluence_score)
        
        logger.debug(f"Analyse MS: Buy Conf={buy_confluence_score:.2f}, Sell Conf={sell_confluence_score:.2f}, " +
                    f"Buy Signal={ms_signals['ms_buy_signal']}, Sell Signal={ms_signals['ms_sell_signal']}, " +
                    f"Ref Level={ms_signals['ms_reference_level']}")
        
        return ms_signals

    def analyze_of_for_trade_signal(self, of_features: Dict) -> Dict:
        """
        Analyse les features Order Flow pour obtenir des signaux de trading.
        """
        of_signals = {
            'of_buy_signal': False,
            'of_sell_signal': False,
            'of_neutral': True,
            'of_strength': 0.0,
            'of_reason': None
        }
        
        if not of_features or self.config.getboolean('order_flow', 'enabled', fallback=False) == False:
            return of_signals
        
        cvd = of_features.get('of_cvd', 0.0)
        absorption_score = of_features.get('of_absorption_score', 0.0)
        trapped_traders_bias = of_features.get('of_trapped_traders_bias', 0.0)
        
        cvd_buy_threshold = self.config.getfloat('order_flow', 'of_cvd_buy_threshold', fallback=10000)
        cvd_sell_threshold = self.config.getfloat('order_flow', 'of_cvd_sell_threshold', fallback=-10000)
        absorption_buy_threshold = self.config.getfloat('order_flow', 'of_absorption_buy_threshold', fallback=0.5)
        absorption_sell_threshold = self.config.getfloat('order_flow', 'of_absorption_sell_threshold', fallback=-0.5)
        of_buy_bias_threshold = self.config.getfloat('order_flow', 'of_buy_bias_threshold', fallback=0.5)
        of_sell_bias_threshold = self.config.getfloat('order_flow', 'of_sell_bias_threshold', fallback=-0.5)
        
        buy_score = 0.0
        sell_score = 0.0
        buy_reasons = []
        sell_reasons = []
        
        if cvd > cvd_buy_threshold:
            buy_score += min(cvd / cvd_buy_threshold, 3.0) * 0.3
            buy_reasons.append(f"CVD={cvd:.0f}")
        elif cvd < cvd_sell_threshold:
            sell_score += min(abs(cvd / cvd_sell_threshold), 3.0) * 0.3
            sell_reasons.append(f"CVD={cvd:.0f}")
        
        if absorption_score > absorption_buy_threshold:
            buy_score += min(absorption_score / absorption_buy_threshold, 2.0) * 0.4
            buy_reasons.append(f"Abs={absorption_score:.2f}")
        elif absorption_score < absorption_sell_threshold:
            sell_score += min(abs(absorption_score / absorption_sell_threshold), 2.0) * 0.4
            sell_reasons.append(f"Abs={absorption_score:.2f}")
        
        if trapped_traders_bias > 0:
            buy_score += min(trapped_traders_bias / of_buy_bias_threshold, 2.0) * 0.3
            buy_reasons.append(f"Trapped={trapped_traders_bias:.2f}")
        elif trapped_traders_bias < 0:
            sell_score += min(abs(trapped_traders_bias / of_sell_bias_threshold), 2.0) * 0.3
            sell_reasons.append(f"Trapped={trapped_traders_bias:.2f}")
        
        of_signals['of_buy_signal'] = buy_score >= 0.6
        of_signals['of_sell_signal'] = sell_score >= 0.6
        of_signals['of_neutral'] = not (of_signals['of_buy_signal'] or of_signals['of_sell_signal'])
        
        of_signals['of_strength'] = max(buy_score, sell_score)
        
        if of_signals['of_buy_signal']:
            of_signals['of_reason'] = "Buy: " + ", ".join(buy_reasons)
        elif of_signals['of_sell_signal']:
            of_signals['of_reason'] = "Sell: " + ", ".join(sell_reasons)
        
        logger.debug(f"Analyse OF: Buy Score={buy_score:.2f}, Sell Score={sell_score:.2f}, " +
                    f"Signal={'Buy' if of_signals['of_buy_signal'] else ('Sell' if of_signals['of_sell_signal'] else 'Neutral')}, " +
                    f"Strength={of_signals['of_strength']:.2f}, Reason={of_signals['of_reason']}")
        
        return of_signals

    def evaluate_setup_quality(
        self,
        ms_signals: Dict,
        of_signals: Dict,
        xgb_probabilities: Optional[np.ndarray],
        pi_confidence_score: float,
        fq_predicted_ratios: Optional[np.ndarray],
        sentiment_score: float,
        ms_relevance_factor: float = 1.0
    ) -> str:
        """
        Évalue la qualité globale du setup (A, B, ou C) basée sur les signaux de tous les modèles.
        """
        if not (ms_signals.get('ms_buy_signal', False) or ms_signals.get('ms_sell_signal', False)):
            return 'NONE'
        
        component_scores = {
            'ms_score': 0.0,
            'of_score': 0.0,
            'xgb_score': 0.0,
            'pi_score': 0.0,
            'fq_score': 0.0,
            'sentiment_score': 0.0
        }
        
        ms_confluence = ms_signals.get('ms_confluence_score', 0.0)
        component_scores['ms_score'] = min(ms_confluence, 5.0) * ms_relevance_factor
        
        if of_signals:
            of_strength = of_signals.get('of_strength', 0.0)
            component_scores['of_score'] = of_strength * 5.0
        
        if xgb_probabilities is not None:
            is_buy_signal = ms_signals.get('ms_buy_signal', False)
            prob_idx = 1 if is_buy_signal else 2
            xgb_prob = xgb_probabilities[prob_idx]
            component_scores['xgb_score'] = min(xgb_prob * 5.0, 5.0)
        
        abs_confidence = abs(pi_confidence_score)
        if (ms_signals.get('ms_buy_signal', False) and pi_confidence_score > 0) or \
           (ms_signals.get('ms_sell_signal', False) and pi_confidence_score < 0):
            component_scores['pi_score'] = min(abs_confidence * 10.0, 5.0)
        
        if fq_predicted_ratios is not None and len(fq_predicted_ratios) >= 3:
            is_buy_signal = ms_signals.get('ms_buy_signal', False)
            q10_q90_diff = abs(fq_predicted_ratios[2] - fq_predicted_ratios[0])
            q50_movement = fq_predicted_ratios[1] - 1.0
            
            if (is_buy_signal and q50_movement > 0) or (not is_buy_signal and q50_movement < 0):
                component_scores['fq_score'] = min(abs(q50_movement) * 20.0 + q10_q90_diff * 10.0, 5.0)
        
        is_buy_signal = ms_signals.get('ms_buy_signal', False)
        if (is_buy_signal and sentiment_score > 0) or (not is_buy_signal and sentiment_score < 0):
            component_scores['sentiment_score'] = min(abs(sentiment_score) * 5.0, 5.0)
        
        total_score = sum(component_scores.values())
        max_possible_score = 30.0
        
        if total_score >= 0.7 * max_possible_score:
            setup_quality = 'A'
        elif total_score >= 0.5 * max_possible_score:
            setup_quality = 'B'
        elif total_score >= 0.3 * max_possible_score:
            setup_quality = 'C'
        else:
            setup_quality = 'NONE'
        
        logger.debug(f"Qualité du setup: {setup_quality} (Score: {total_score:.1f}/{max_possible_score})")
        logger.debug(f"Scores par composant: MS={component_scores['ms_score']:.1f}, " +
                    f"OF={component_scores['of_score']:.1f}, XGB={component_scores['xgb_score']:.1f}, " +
                    f"Pi={component_scores['pi_score']:.1f}, FQ={component_scores['fq_score']:.1f}, " +
                    f"Sent={component_scores['sentiment_score']:.1f}")
        
        return setup_quality

    def generate_trade_decision(self, symbol_market: str,
                                current_ratings: Dict[str, float],
                                latest_price: float,
                                sentiment_score: float,
                                xgb_probabilities: Optional[np.ndarray],
                                fq_predicted_ratios: Optional[np.ndarray],
                                market_structure_features: Dict,
                                order_flow_features: Dict,
                                atr_value: Optional[float],
                                current_position_qty: float = 0.0,
                                account_balance: float = 1000.0,
                                daily_pnl: float = 0.0,
                                market_regime: str = "Indeterminate", # Added market_regime
                                sentiment_data: Optional[Dict] = None # Added full sentiment data
                                ) -> Dict:
        """
        Combine les outputs des modèles ML, les Pi-Ratings, le sentiment, la structure de marché et l'order flow
        pour prendre une décision de trading. Implémente la logique Dynamic Risk Management (DRM) pour le sizing.
        """
        action = 'HOLD'
        target_qty = 0.0
        tp_price = None
        sl_price = None
        position_side_to_open = None
        
        setup_quality = 'NONE'
        risk_multiplier = 1.0
        exit_strategy = {}

        if self.config is None:
            logger.error("ConfigManager non fourni à generate_trade_decision. Impossible de récupérer les paramètres.")
            return {
                'action': 'HOLD', 'target_qty': 0.0, 'tp_price': None, 'sl_price': None,
                'position_side_to_open': None, 'confidence_score': 0.0, 'confidence_trend': 'neutral',
                'setup_quality': 'NONE', 'risk_multiplier': 0.0, 'exit_strategy': {},
                'ms_signals': {}, 'of_signals': {}
            }

        is_long = current_position_qty > 0
        is_short = current_position_qty < 0

        r_h = current_ratings.get('R_H', 0.0)
        r_a = current_ratings.get('R_A', 0.0)
        confidence_score = self.indicators_manager.calculate_confidence_score(r_h, r_a)

        ms_signals = self.analyze_ms_for_trade_signal(market_structure_features, latest_price)
        of_signals = self.analyze_of_for_trade_signal(order_flow_features)

        ms_relevance_factor = 1.0
        ms_formation_atr_normalized = market_structure_features.get('ms_formation_atr_normalized')

        current_normalized_atr = None
        if atr_value is not None and latest_price > 0:
            current_normalized_atr = atr_value / latest_price

        if current_normalized_atr is not None and ms_formation_atr_normalized is not None and ms_formation_atr_normalized > 1e-9:
            vol_ratio = current_normalized_atr / ms_formation_atr_normalized
            
            vol_ratio_high_threshold = self.config.getfloat('strategy_hybrid', 'ms_relevance_vol_ratio_high', fallback=2.0)
            vol_ratio_low_threshold = self.config.getfloat('strategy_hybrid', 'ms_relevance_vol_ratio_low', fallback=0.5)
            penalty_high_vol_mismatch = self.config.getfloat('strategy_hybrid', 'ms_relevance_penalty_high_vol', fallback=0.7)
            penalty_low_vol_mismatch = self.config.getfloat('strategy_hybrid', 'ms_relevance_penalty_low_vol', fallback=0.85)

            if vol_ratio > vol_ratio_high_threshold:
                ms_relevance_factor *= penalty_high_vol_mismatch
                logger.debug(f"MS Relevance: Current vol ({current_normalized_atr:.4f}) much higher than formation vol ({ms_formation_atr_normalized:.4f}). Ratio: {vol_ratio:.2f}. Factor: {ms_relevance_factor:.2f}")
            elif vol_ratio < vol_ratio_low_threshold:
                ms_relevance_factor *= penalty_low_vol_mismatch
                logger.debug(f"MS Relevance: Current vol ({current_normalized_atr:.4f}) much lower than formation vol ({ms_formation_atr_normalized:.4f}). Ratio: {vol_ratio:.2f}. Factor: {ms_relevance_factor:.2f}")
            else:
                logger.debug(f"MS Relevance: Volatility context is comparable. Current_norm_ATR: {current_normalized_atr:.4f}, Formation_norm_ATR: {ms_formation_atr_normalized:.4f}, Ratio: {vol_ratio:.2f}")
        elif ms_formation_atr_normalized is None:
            logger.debug("MS Relevance: ms_formation_atr_normalized not available. Using default factor 1.0.")

        try:
            xgb_buy_prob_threshold = self.config.getfloat('strategy_hybrid', 'xgb_buy_prob_threshold', fallback=0.6)
            pi_rating_diff_threshold_buy = self.config.getfloat('strategy_hybrid', 'pi_rating_diff_threshold_buy', fallback=0.1)
            pi_rating_ratio_threshold_buy = self.config.getfloat('strategy_hybrid', 'pi_rating_ratio_threshold_buy', fallback=1.2)
            pi_confidence_threshold_buy = self.config.getfloat('strategy_hybrid', 'pi_confidence_threshold_buy', fallback=0.3)
            fq_q50_multiplier_buy = self.config.getfloat('strategy_hybrid', 'fq_q50_multiplier_buy', fallback=1.005)
            fq_q10_multiplier_buy = self.config.getfloat('strategy_hybrid', 'fq_q10_multiplier_buy', fallback=1.001)
            sentiment_min_buy_threshold = self.config.getfloat('strategy_hybrid', 'sentiment_min_buy_threshold', fallback=0.1)

            xgb_sell_prob_threshold = self.config.getfloat('strategy_hybrid', 'xgb_sell_prob_threshold', fallback=0.6)
            pi_rating_diff_threshold_sell = self.config.getfloat('strategy_hybrid', 'pi_rating_diff_threshold_sell', fallback=0.1)
            pi_rating_ratio_threshold_sell = self.config.getfloat('strategy_hybrid', 'pi_rating_ratio_threshold_sell', fallback=1.2)
            pi_confidence_threshold_sell = self.config.getfloat('strategy_hybrid', 'pi_confidence_threshold_sell', fallback=-0.3)
            fq_q50_multiplier_sell = self.config.getfloat('strategy_hybrid', 'fq_q50_multiplier_sell', fallback=0.995)
            fq_q90_multiplier_sell = self.config.getfloat('strategy_hybrid', 'fq_q90_multiplier_sell', fallback=0.999)
            sentiment_max_sell_threshold = self.config.getfloat('strategy_hybrid', 'sentiment_max_sell_threshold', fallback=-0.1)

            sl_distance_percent = self.config.getfloat('strategy_hybrid', 'sl_distance_percent', fallback=0.015)
            tp_sl_ratio = self.config.getfloat('strategy_hybrid', 'tp_sl_ratio', fallback=self.indicators_manager.PHI)
            
            setup_size_multiplier_a = self.config.getfloat('dynamic_risk', 'setup_size_multiplier_a', fallback=1.0)
            setup_size_multiplier_b = self.config.getfloat('dynamic_risk', 'setup_size_multiplier_b', fallback=0.6)
            setup_size_multiplier_c = self.config.getfloat('dynamic_risk', 'setup_size_multiplier_c', fallback=0.3)
            min_risk_multiplier = self.config.getfloat('dynamic_risk', 'min_risk_multiplier', fallback=0.1)
            
            move_to_be_profit_percent = self.config.getfloat('dynamic_risk', 'move_to_be_profit_percent', fallback=0.005)
            trailing_stop_profit_percent_start = self.config.getfloat('dynamic_risk', 'trailing_stop_profit_percent_start', fallback=0.01)
            trailing_stop_distance_percent = self.config.getfloat('dynamic_risk', 'trailing_stop_distance_percent', fallback=0.005)

        except (ValueError, NoSectionError, NoOptionError) as e:
            logger.error(f"Erreur de configuration dans strategy_hybrid/dynamic_risk: {e}. Utilisation des fallbacks.")
            return {
                'action': 'HOLD', 'target_qty': 0.0, 'tp_price': None, 'sl_price': None,
                'position_side_to_open': None, 'confidence_score': confidence_score, 'confidence_trend': 'neutral',
                'setup_quality': 'NONE', 'risk_multiplier': 0.0, 'exit_strategy': {},
                'ms_signals': ms_signals, 'of_signals': of_signals
            }

        logger.debug(f"Décision trading {symbol_market}: R_H={r_h:.4f}, R_A={r_a:.4f}, Confidence={confidence_score:.4f}, Sentiment={sentiment_score:.4f}, Pos={current_position_qty:.8f}")
        if xgb_probabilities is not None:
            logger.debug(f"XGBoost Probas: Hold={xgb_probabilities[0]:.4f}, Buy={xgb_probabilities[1]:.4f}, Sell={xgb_probabilities[2]:.4f}")
        
        fq_pred_q10_price, fq_pred_q50_price, fq_pred_q90_price = None, None, None
        if fq_predicted_ratios is not None:
            logger.debug(f"FQ Predicted Ratios: {fq_predicted_ratios.tolist()}")
            fq_predicted_prices = fq_predicted_ratios * latest_price
            logger.debug(f"FQ Predicted Prices (based on {latest_price:.4f}): {fq_predicted_prices.tolist()}")
            if len(fq_predicted_prices) >= 3:
                fq_pred_q10_price = fq_predicted_prices[0]
                fq_pred_q50_price = fq_predicted_prices[1]
                fq_pred_q90_price = fq_predicted_prices[self.config.getint('futurequant','quantiles',fallback='0.1,0.5,0.9').count(',') if self.config.get('futurequant','quantiles',fallback='0.1,0.5,0.9').count(',') < len(fq_predicted_prices) else -1]
            else:
                logger.warning("Moins de 3 quantiles prédits par FQ, vérification Q10/Q50/Q90 affectée.")

        is_xgb_buy_biased = (xgb_probabilities is not None) and (xgb_probabilities[1] > xgb_buy_prob_threshold)
        is_pi_ratings_bullish = (r_h > r_a and (r_h - r_a > pi_rating_diff_threshold_buy or 
                                (r_a > 1e-9 and r_h / r_a > pi_rating_ratio_threshold_buy))) or \
                                (confidence_score > pi_confidence_threshold_buy)
        is_fq_bullish = (fq_pred_q50_price is not None and fq_pred_q10_price is not None) and \
                       (fq_pred_q50_price > latest_price * fq_q50_multiplier_buy) and \
                       (fq_pred_q10_price > latest_price * fq_q10_multiplier_buy)
        is_sentiment_bullish = sentiment_score >= sentiment_min_buy_threshold
        
        is_ms_buy_signal = ms_signals.get('ms_buy_signal', False)
        is_of_buy_signal = of_signals.get('of_buy_signal', False)

        logger.debug(f"Buy Conditions: XGB({is_xgb_buy_biased}), PI({is_pi_ratings_bullish}), " + 
                    f"FQ({is_fq_bullish}), Sent({is_sentiment_bullish}), MS({is_ms_buy_signal}), OF({is_of_buy_signal})")

        is_xgb_sell_biased = (xgb_probabilities is not None) and (xgb_probabilities[2] > xgb_sell_prob_threshold)
        is_pi_ratings_bearish = (r_a > r_h and (r_a - r_h > pi_rating_diff_threshold_sell or 
                                (r_h > 1e-9 and r_a / r_h > pi_rating_ratio_threshold_sell))) or \
                                (confidence_score < pi_confidence_threshold_sell)
        is_fq_bearish = (fq_pred_q50_price is not None and fq_pred_q90_price is not None) and \
                        (fq_pred_q50_price < latest_price * fq_q50_multiplier_sell) and \
                        (fq_pred_q90_price < latest_price * fq_q90_multiplier_sell)
        is_sentiment_bearish = sentiment_score <= sentiment_max_sell_threshold
        
        is_ms_sell_signal = ms_signals.get('ms_sell_signal', False)
        is_of_sell_signal = of_signals.get('of_sell_signal', False)

        logger.debug(f"Sell Conditions: XGB({is_xgb_sell_biased}), PI({is_pi_ratings_bearish}), " +
                    f"FQ({is_fq_bearish}), Sent({is_sentiment_bearish}), MS({is_ms_sell_signal}), OF({is_of_sell_signal})")

        pi_history = self.indicators_manager.get_ratings_history(symbol_market, limit=5)
        confidence_trending_up = False
        confidence_trending_down = False
        min_history_for_trend = 3
        
        if len(pi_history) >= min_history_for_trend:
            recent_confidence = [entry.get('confidence', 0) for entry in pi_history[-min_history_for_trend:]]
            if len(recent_confidence) == min_history_for_trend :
                 confidence_slope = (recent_confidence[-1] - recent_confidence[0]) / (min_history_for_trend - 1) if min_history_for_trend > 1 else 0
                 confidence_trending_up = confidence_slope > 0.05
                 confidence_trending_down = confidence_slope < -0.05
            
            logger.debug(f"Pi-Ratings confidence trend: last values={recent_confidence}, slope={confidence_slope if 'confidence_slope' in locals() else 'N/A'}, "
                        f"up={confidence_trending_up}, down={confidence_trending_down}")

        setup_quality = self.evaluate_setup_quality(
            ms_signals, of_signals, xgb_probabilities, 
            confidence_score, fq_predicted_ratios, sentiment_score,
            ms_relevance_factor=ms_relevance_factor
        )
        
        if is_long:
            logger.debug("Actuellement en position LONG.")
            model_sell_signals = sum([is_xgb_sell_biased, is_pi_ratings_bearish, is_fq_bearish, confidence_trending_down])
            technical_sell_signals = sum([is_ms_sell_signal, is_of_sell_signal, is_sentiment_bearish])
            
            close_long_signal = (model_sell_signals >= 2) or \
                                (technical_sell_signals >= 2) or \
                                (is_xgb_sell_biased and (is_ms_sell_signal or is_of_sell_signal)) or \
                                (is_pi_ratings_bearish and (is_ms_sell_signal or is_of_sell_signal))

            if close_long_signal:
                logger.info(f"Signal de CLOTURE LONG multi-facteur détecté pour {symbol_market}. " + 
                           f"Models({model_sell_signals}/4), Technical({technical_sell_signals}/3)")
                action = 'SELL_TO_CLOSE_LONG'
                target_qty = current_position_qty 
            else:
                action = 'HOLD'
                logger.debug("Aucun signal de clôture LONG multi-facteur. Maintien position.")

        elif current_position_qty == 0:
            logger.debug("Actuellement plat.")
            
            if setup_quality == 'A':
                risk_multiplier = setup_size_multiplier_a
            elif setup_quality == 'B':
                risk_multiplier = setup_size_multiplier_b
            elif setup_quality == 'C':
                risk_multiplier = setup_size_multiplier_c
            else:
                risk_multiplier = min_risk_multiplier
            
            model_buy_signals = sum([is_xgb_buy_biased, is_pi_ratings_bullish, is_fq_bullish, confidence_trending_up])
            technical_buy_signals = sum([is_ms_buy_signal, is_of_buy_signal, is_sentiment_bullish])
            
            entry_signal_base = (model_buy_signals >= 2 and technical_buy_signals >= 2) or \
                                (model_buy_signals >= 3 and technical_buy_signals >= 1)
            
            entry_signal_triggered_by_ms_of = entry_signal_base and (is_ms_buy_signal or is_of_buy_signal)
            
            allow_trade_on_quality = setup_quality in ['A', 'B', 'C']

            if entry_signal_triggered_by_ms_of and allow_trade_on_quality:
                logger.info(f"Signal d'ACHAT MULTI-FACTEUR détecté pour {symbol_market}. " + 
                           f"Models({model_buy_signals}/4), Technical({technical_buy_signals}/3), " +
                           f"Quality={setup_quality}, Risk Mult={risk_multiplier:.2f}")
                
                action = 'BUY'
                position_side_to_open = 'long'
                
                ms_reference_level = ms_signals.get('ms_reference_level')
                if ms_reference_level is not None and ms_reference_level > 0 and (is_ms_buy_signal or is_ms_sell_signal) :
                    sl_candidate = ms_reference_level * (0.995 if is_ms_buy_signal else 1.005)
                    if (is_ms_buy_signal and sl_candidate < latest_price) or (not is_ms_buy_signal and sl_candidate > latest_price):
                        sl_price = sl_candidate
                        sl_distance = abs(latest_price - sl_price)
                        tp_price = latest_price + (sl_distance * tp_sl_ratio) if is_ms_buy_signal else latest_price - (sl_distance * tp_sl_ratio)
                        logger.debug(f"Calcul TP/SL basé sur niveau MS: Ref={ms_reference_level:.4f}, " +
                                   f"TP={tp_price:.4f}, SL={sl_price:.4f}")
                    else:
                        tp_price, sl_price = self.calculate_fib_tp_sl_prices(
                            latest_price, True, sl_distance_percent, tp_sl_ratio
                        )
                        logger.debug(f"Niveau MS non logique pour SL. Calcul TP/SL standard pour BUY: TP={tp_price}, SL={sl_price}")

                else:
                    use_dynamic_sl_tp = self.config.getboolean('strategy_hybrid', 'use_dynamic_sl_tp', fallback=False)
                    dynamic_sl_tp_method = self.config.get('strategy_hybrid', 'dynamic_sl_tp_method', fallback='atr')
                    
                    tp_sl_ratio = self.config.getfloat('strategy_hybrid', 'tp_sl_ratio', fallback=1.618)

                    if use_dynamic_sl_tp and dynamic_sl_tp_method == 'atr':
                        if atr_value is not None and atr_value > 0:
                            atr_sl_multiplier = self.config.getfloat('strategy_hybrid', 'atr_sl_multiplier', fallback=1.5)
                            logger.info(f"[{symbol_market}] Using DYNAMIC ATR-based SL/TP. Price: {latest_price:.4f}, ATR: {atr_value:.4f}, SL Mult: {atr_sl_multiplier}, TP Ratio: {tp_sl_ratio}")
                            tp_price, sl_price = self.calculate_atr_tp_sl_prices(
                                entry_price=latest_price, 
                                is_long=True, 
                                atr_value=atr_value, 
                                atr_sl_multiplier=atr_sl_multiplier, 
                                tp_sl_ratio=tp_sl_ratio
                            )
                        else:
                            logger.warning(f"[{symbol_market}] Dynamic ATR SL/TP requested but ATR value is missing or invalid ({atr_value}). Falling back to fixed SL/TP if configured, or no SL/TP.")
                            use_dynamic_sl_tp = False
                    
                    if not use_dynamic_sl_tp:
                        sl_distance_percent = self.config.getfloat('strategy_hybrid', 'sl_distance_percent', fallback=None)
                        if sl_distance_percent is not None and sl_distance_percent > 0:
                            logger.info(f"[{symbol_market}] Using FIXED percentage-based SL/TP. Price: {latest_price:.4f}, SL %: {sl_distance_percent*100:.2f}%, TP Ratio: {tp_sl_ratio}")
                            tp_price, sl_price = self.calculate_fib_tp_sl_prices(
                                entry_price=latest_price, 
                                is_long=True, 
                                sl_distance_percent=sl_distance_percent, 
                                tp_sl_ratio=tp_sl_ratio
                            )
                        else:
                            logger.warning(f"[{symbol_market}] Fixed SL/TP requested, but sl_distance_percent not configured or invalid. No SL/TP will be set.")
                            tp_price, sl_price = None, None

            else:
                action = 'HOLD'
                logger.debug(f"Aucun signal d'achat multi-facteur ou qualité de setup insuffisante (Qualité: {setup_quality}).")

        return {
            'action': action,
            'target_qty': target_qty,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'position_side_to_open': position_side_to_open,
            'confidence_score': confidence_score,
            'confidence_trend': 'up' if confidence_trending_up else ('down' if confidence_trending_down else 'neutral'),
            'setup_quality': setup_quality,
            'risk_multiplier': risk_multiplier,
            'exit_strategy': exit_strategy,
            'ms_signals': ms_signals,
            'of_signals': of_signals, # Added missing comma
        }