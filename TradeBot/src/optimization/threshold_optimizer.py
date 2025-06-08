# threshold_optimizer.py
import pandas as pd
import numpy as np
import logging
from configparser import ConfigParser
import json
import os
import itertools # For Grid Search
import random    # For Random Search
from typing import Dict, List, Any, Tuple, Optional

# Project Modules
from backtester import Backtester 
import training_pipeline # For loading prepared data and models
import feature_engineer # For scaler loading (if not handled by training_pipeline._load_models)
import model_xgboost    # For XGBoost model loading
import model_futurequant # For FQ model loading

logger = logging.getLogger(__name__)
CONFIG_FILE = 'config.ini' # Default config file

# --- Helper function to load all necessary components for backtesting ---
def load_backtesting_components(config: ConfigParser) -> Tuple[Optional[Dict[str, pd.DataFrame]], Optional[Dict[str, Any]]]:
    """
    Loads prepared data and ML models needed for backtesting.
    """
    logger.info("Loading backtesting components...")
    prepared_data = training_pipeline.load_prepared_data_for_analysis(config)
    if not prepared_data:
        logger.error("Failed to load prepared data for optimization.")
        return None, None

    loaded_models = {}
    try:
        # Initialize modules to load models/scalers
        feature_engineer.initialize_feature_scalers(config)
        loaded_models['scaler'] = feature_engineer.get_scaler('feature_scaler') 
        # ^ Assuming 'feature_scaler' is the name of the main scaler saved by training_pipeline

        model_xgboost.initialize_xgboost_model(config)
        loaded_models['xgb_model'] = model_xgboost.get_model()
        
        model_futurequant.initialize_futurequant_model(config)
        loaded_models['fq_model'] = model_futurequant.get_model()

        training_params_file = config.get('training', 'training_params_file', fallback='training_params.json')
        if os.path.exists(training_params_file):
            with open(training_params_file, 'r') as f:
                loaded_models['training_params'] = json.load(f)
                logger.info(f"Training parameters loaded from {training_params_file}")
        else:
            logger.warning(f"Training params file {training_params_file} not found. Model input features may be unknown.")
            loaded_models['training_params'] = {}

        if not loaded_models.get('scaler'):
            logger.error("Scaler could not be loaded. Essential for backtesting.")
            return prepared_data, None # Allow running without full models but not without scaler
        if not loaded_models.get('xgb_model'):
            logger.warning("XGBoost model could not be loaded.")
        if not loaded_models.get('fq_model'):
            logger.warning("FutureQuant model could not be loaded.")

    except Exception as e:
        logger.error(f"Error loading models/parameters for optimization: {e}", exc_info=True)
        return prepared_data, None
        
    return prepared_data, loaded_models


class ThresholdOptimizer:
    def __init__(self,
                 config: ConfigParser,
                 prepared_data: Dict[str, pd.DataFrame],
                 loaded_models: Dict[str, Any],
                 parameter_space: Dict[str, List[Any]], 
                 optimization_metric: str = 'sharpe_ratio',
                 maximize_metric: bool = True):

        self.config = config
        self.prepared_data = prepared_data
        self.loaded_models = loaded_models
        self.parameter_space = parameter_space
        self.optimization_metric = optimization_metric
        self.maximize_metric = maximize_metric
        
        self.results: List[Dict[str, Any]] = [] 

    def _objective_function(self, current_strategy_params: Dict[str, Any]) -> float:
        """
        Runs a single backtest with the given strategy parameters and returns the target metric.
        """
        # Ensure base config is not accidentally modified if passed around elsewhere
        temp_config = ConfigParser()
        temp_config.read_dict(self.config) 

        # Override config sections with current_strategy_params for strategy_hybrid to pick up
        # This assumes current_strategy_params keys might include section names like 'strategy_hybrid/param_name'
        # or are flat and applied to a default section like 'strategy_params_override'
        # For simplicity, let's assume strategy_params given to Backtester directly handles this.

        logger.info(f"Running backtest with parameters: {current_strategy_params}")
        
        try:
            backtester_instance = Backtester(
                prepared_data=self.prepared_data,
                config=temp_config, 
                strategy_params=current_strategy_params, 
                loaded_models=self.loaded_models
            )
            backtest_results = backtester_instance.run_backtest()
        except Exception as e:
            logger.error(f"Exception during backtest run with params {current_strategy_params}: {e}", exc_info=True)
            return -float('inf') if self.maximize_metric else float('inf')

        self.results.append({
            'params': current_strategy_params.copy(),
            'metrics': backtest_results.copy() if backtest_results else {}
        })

        metric_value = backtest_results.get(self.optimization_metric) if backtest_results else None
        if metric_value is None or (isinstance(metric_value, float) and np.isnan(metric_value)):
            logger.warning(f"Opt. metric '{self.optimization_metric}' not found or NaN. Params: {current_strategy_params}")
            return -float('inf') if self.maximize_metric else float('inf')
        
        logger.info(f"Params: {current_strategy_params} -> {self.optimization_metric}: {metric_value:.4f}")
        return float(metric_value)

    def grid_search(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        logger.info("Starting Grid Search...")
        param_names = list(self.parameter_space.keys())
        param_value_lists = list(self.parameter_space.values())
        
        best_score = -float('inf') if self.maximize_metric else float('inf')
        best_params = None
        best_metrics_package = None # To store the whole metrics dict for the best run
        
        total_combinations = np.prod([len(v) for v in param_value_lists]) if param_value_lists else 0
        if total_combinations == 0:
            logger.warning("No parameter combinations to test in Grid Search.")
            return None, None
        logger.info(f"Total parameter combinations for Grid Search: {total_combinations}")
        
        count = 0
        for values_combination in itertools.product(*param_value_lists):
            count += 1
            current_params_iteration = dict(zip(param_names, values_combination))
            score = self._objective_function(current_params_iteration)
            
            if (self.maximize_metric and score > best_score) or \
               (not self.maximize_metric and score < best_score):
                best_score = score
                best_params = current_params_iteration.copy()
                # Find the full metrics for the best_params from self.results
                for res in reversed(self.results): 
                    if res['params'] == best_params:
                        best_metrics_package = res['metrics']
                        break
            logger.info(f"Grid Search Progress: {count}/{total_combinations} | Current Best {self.optimization_metric}: {best_score:.4f}")

        logger.info("Grid Search finished.")
        if best_params:
            logger.info(f"Best parameters found: {best_params}")
            logger.info(f"Best {self.optimization_metric}: {best_score:.4f}")
            # logger.info(f"Full metrics for best parameters: {best_metrics_package}")
        else:
            logger.warning("Grid Search did not find any suitable parameters.")
            
        return best_params, best_metrics_package

    def random_search(self, n_iterations: int) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        logger.info(f"Starting Random Search for {n_iterations} iterations...")
        if not self.parameter_space or n_iterations == 0:
            logger.warning("No parameter space or zero iterations for Random Search.")
            return None, None
            
        param_names = list(self.parameter_space.keys())
        best_score = -float('inf') if self.maximize_metric else float('inf')
        best_params = None
        best_metrics_package = None

        for i in range(n_iterations):
            current_params_iteration = {}
            for name in param_names:
                if not self.parameter_space[name]: # Skip if list of choices is empty
                    logger.warning(f"Parameter '{name}' has no values in parameter_space. Skipping.")
                    continue
                current_params_iteration[name] = random.choice(self.parameter_space[name])
            
            if not current_params_iteration: # If all params had empty choices
                logger.error("Could not generate any parameter set for random search iteration.")
                continue

            score = self._objective_function(current_params_iteration)
            
            if (self.maximize_metric and score > best_score) or \
               (not self.maximize_metric and score < best_score):
                best_score = score
                best_params = current_params_iteration.copy()
                for res in reversed(self.results):
                    if res['params'] == best_params:
                        best_metrics_package = res['metrics']
                        break
            logger.info(f"Random Search Progress: {i+1}/{n_iterations} | Current Best {self.optimization_metric}: {best_score:.4f}")

        logger.info("Random Search finished.")
        if best_params:
            logger.info(f"Best parameters found: {best_params}")
            logger.info(f"Best {self.optimization_metric}: {best_score:.4f}")
        else:
            logger.warning("Random Search did not find any suitable parameters.")

        return best_params, best_metrics_package

    def save_optimization_results(self, filepath: str = "optimization_results.json"):
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=4, default=str) 
            logger.info(f"Optimization results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving optimization results: {e}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("threshold_optimizer.log"), logging.StreamHandler()])
    
    config = ConfigParser()
    if not config.read(CONFIG_FILE):
        logger.critical(f"Configuration file {CONFIG_FILE} not found. Exiting.")
        return

    parameter_space_definition = {
        'xgb_buy_prob_threshold': [0.55, 0.6, 0.65],
        'xgb_sell_prob_threshold': [0.55, 0.6, 0.65],
        'pi_confidence_threshold_buy': [0.25, 0.3, 0.35],
        'pi_confidence_threshold_sell': [-0.25, -0.3, -0.35],
        'sentiment_min_buy_threshold': [0.05, 0.1, 0.15],
        'sentiment_max_sell_threshold': [-0.05, -0.1, -0.15],
        'move_to_be_profit_percent': [0.005, 0.0075, 0.01],
        'trailing_stop_profit_percent_start': [0.01, 0.015],
        'trailing_stop_distance_percent': [0.005, 0.0075],
        # For MS/OF, ensure strategy_hybrid.py can accept these via strategy_params
        # e.g., add arguments to analyze_ms_for_trade_signal or have it use get_strat_param
        # 'ms_buy_confluence_min': [2.0, 2.5],
        # 'of_buy_score_min': [0.5, 0.6]
    }

    prepared_data, loaded_models_components = load_backtesting_components(config)

    if not prepared_data or not loaded_models_components or not loaded_models_components.get('scaler') or not loaded_models_components.get('training_params'):
        logger.critical("Failed to load essential components (data, scaler, training_params) for optimization. Exiting.")
        return

    optimizer = ThresholdOptimizer(
        config=config,
        prepared_data=prepared_data,
        loaded_models=loaded_models_components,
        parameter_space=parameter_space_definition,
        optimization_metric='sharpe_ratio', 
        maximize_metric=True
    )

    logger.info("Starting optimization process...")
    # best_params_found, best_metrics_found = optimizer.grid_search()
    best_params_found, best_metrics_found = optimizer.random_search(n_iterations=20) # More iterations for random

    if best_params_found:
        logger.info("\n--- Optimization Complete ---")
        logger.info(f"Best Parameters: {best_params_found}")
        if best_metrics_found:
            logger.info(f"Metrics for Best Parameters (full package): {best_metrics_found.get(optimizer.optimization_metric)}")
            # logger.info(f"Full metrics dict: {best_metrics_found}")
    else:
        logger.info("\n--- Optimization Complete ---")
        logger.info("No optimal parameters were found or the process was interrupted.")

    optimizer.save_optimization_results("threshold_optimization_run_results.json")

if __name__ == "__main__":
    main() 