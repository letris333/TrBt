#!/usr/bin/env python3
"""
Parameter Optimizer for Trading System

This script optimizes strategy parameters using the backtester to find optimal configurations.
It supports various optimization methods including grid search, random search, and Bayesian optimization.
"""

import json
import logging
import argparse
import os
import time
import random
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
from configparser import ConfigParser, NoSectionError, NoOptionError # Ajouté NoSectionError, NoOptionError
import matplotlib.pyplot as plt

# Import trading system modules
import trading_analyzer # Contient Backtester fusionné et fonctions de chargement
# Backtester est implicitement utilisé via trading_analyzer.execute_backtest

# Setup logging
# Le logger est déjà configuré dans trading_analyzer, mais on peut en avoir un spécifique ici
# ou s'assurer que le logging de trading_analyzer est appelé.
# Pour l'instant, on garde un logger spécifique pour l'optimiseur.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parameter_optimizer.log", mode='a'), # Append mode
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("parameter_optimizer")

# Default configuration file
CONFIG_FILE = 'config.ini'

class ParameterOptimizer:
    """
    Parameter Optimizer class that runs the optimization process for trading strategy parameters.
    Supports grid search, random search, and Bayesian optimization methods.
    """
    def __init__(self,
                 config_file: str = CONFIG_FILE,
                 optimization_method: str = 'grid',
                 num_runs: int = 100, # For random/Bayesian
                 save_results: bool = True,
                 parallel: bool = False, # Placeholder, implémentation de parallélisation non fournie
                 parameter_spaces_override: Optional[Dict[str, Any]] = None, # Pour passer l'espace de TO
                 target_metric: Optional[str] = None, # Remplacera celui de la config si fourni
                 maximize_metric: bool = True # De TO
                 ):
        """
        Initialize the Parameter Optimizer.
        """
        self.config_file = config_file
        self.optimization_method = optimization_method
        self.num_runs = num_runs
        self.save_results = save_results
        self.parallel = parallel
        self.parameter_spaces_override = parameter_spaces_override
        self.maximize_metric = maximize_metric

        # Load configuration
        self.config = ConfigParser()
        self.config.read(config_file)
        if not self.config.sections():
            logger.critical(f"Le fichier de configuration '{config_file}' est vide ou non trouvé.")
            raise FileNotFoundError(f"Configuration file '{config_file}' not found or empty.")

        # Load trading system components (données, modèles)
        logger.info("Chargement des composants du système de trading pour l'optimisation...")
        # Utiliser la fonction de trading_analyzer pour charger les composants
        # trading_analyzer.setup_logging() # Configurer le logging du module analyzer si besoin
        self.components = trading_analyzer.load_all_components_for_analysis(config_file)
        if not self.components or not self.components.get('prepared_data') or not self.components.get('loaded_models'):
            logger.critical("Échec du chargement des composants nécessaires (données/modèles). Optimisation impossible.")
            raise ValueError("Failed to load necessary components for optimization.")

        # Parameter search spaces
        self.parameter_spaces: Dict[str, Any] = {} # Sera rempli par _load_parameter_spaces
        self._load_parameter_spaces() # Charge depuis config OU utilise l'override

        # Results storage
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_performance: Optional[float] = None
        self.best_metrics_package: Optional[Dict[str, Any]] = None


        # Metrics to optimize
        self.target_metric = target_metric if target_metric is not None else \
                             self.config.get('parameter_optimizer', 'target_metric', fallback='sharpe_ratio')

        logger.info(f"Parameter Optimizer initialisé avec la méthode : {self.optimization_method}")
        logger.info(f"Métrique cible pour l'optimisation : {self.target_metric} (Maximiser: {self.maximize_metric})")

    def _load_parameter_spaces(self):
        """
        Load parameter search spaces.
        Priority: parameter_spaces_override, then config file.
        """
        if self.parameter_spaces_override:
            logger.info("Utilisation de parameter_spaces_override fourni programmatiquement.")
            # Convertir le format de TO (liste de valeurs) au format attendu par PO si nécessaire
            # PO s'attend à {'type': 'categorical_xxx', 'values': [...]} pour les listes
            # ou {'type': 'float'/'int', 'range': [min, max]}
            # TO fournit directement {'param_name': [val1, val2, val3]}
            # Pour l'instant, on suppose que si override est fourni, il est au format attendu par PO,
            # ou on adapte _generate_grid_combinations et _generate_random_params.
            # Simplifions : si override, on assume qu'il est au format de liste de valeurs.
            temp_spaces = {}
            for param, values in self.parameter_spaces_override.items():
                if not isinstance(values, list):
                    logger.warning(f"Override pour '{param}' n'est pas une liste. Ignoré.")
                    continue
                if not values:
                    logger.warning(f"Override pour '{param}' a une liste vide. Ignoré.")
                    continue
                
                # Détecter le type à partir du premier élément
                first_val = values[0]
                if isinstance(first_val, float):
                    temp_spaces[param] = {'type': 'categorical_float', 'values': values}
                elif isinstance(first_val, int):
                    temp_spaces[param] = {'type': 'categorical_int', 'values': values}
                elif isinstance(first_val, str):
                    temp_spaces[param] = {'type': 'categorical_str', 'values': values}
                else:
                    logger.warning(f"Type non supporté pour les valeurs de '{param}' dans l'override. Ignoré.")
                    continue
            self.parameter_spaces = temp_spaces

        elif 'parameter_spaces' in self.config.sections():
            logger.info("Chargement de parameter_spaces depuis le fichier de configuration.")
            for param, value_range_spec in self.config.items('parameter_spaces'):
                try:
                    # Logique de parsing de PO (gère '..', ',', type fixed)
                    if '..' in value_range_spec:
                        start_str, end_str = value_range_spec.split('..')
                        is_float = '.' in start_str or '.' in end_str
                        start_val = float(start_str) if is_float else int(start_str)
                        end_val = float(end_str) if is_float else int(end_str)
                        self.parameter_spaces[param] = {
                            'type': 'float' if is_float else 'int',
                            'range': [start_val, end_val]
                        }
                    elif ',' in value_range_spec:
                        values_str = [v.strip() for v in value_range_spec.split(',')]
                        try:
                            # Essayer de convertir en float, puis en int si possible
                            parsed_values_float = [float(v) for v in values_str]
                            if all(v.is_integer() for v in parsed_values_float):
                                parsed_values_int = [int(v) for v in parsed_values_float]
                                self.parameter_spaces[param] = {'type': 'categorical_int', 'values': parsed_values_int}
                            else:
                                self.parameter_spaces[param] = {'type': 'categorical_float', 'values': parsed_values_float}
                        except ValueError: # Garder comme string si la conversion échoue
                            self.parameter_spaces[param] = {'type': 'categorical_str', 'values': values_str}
                    else: # Valeur unique (fixée)
                        try:
                            val = float(value_range_spec)
                            final_val = int(val) if val.is_integer() else val
                            self.parameter_spaces[param] = {'type': 'fixed', 'value': final_val}
                        except ValueError:
                            self.parameter_spaces[param] = {'type': 'fixed', 'value': value_range_spec}
                except Exception as e:
                    logger.error(f"Erreur de parsing de l'espace des paramètres pour {param}: {e}")
        else:
            logger.error("Aucune section 'parameter_spaces' dans config ET aucun override fourni.")
            # Lever une exception ou retourner avec un espace vide ?
            # Pour l'instant, on continue avec un espace vide, ce qui mènera à un échec plus tard.

        if self.parameter_spaces:
            logger.info(f"Espaces de paramètres chargés ({len(self.parameter_spaces)}):")
            for p, s in self.parameter_spaces.items(): logger.info(f"  - {p}: {s}")
        else:
            logger.warning("Aucun espace de paramètres n'a été chargé pour l'optimisation.")

    def _objective_function(self, current_strategy_params: Dict[str, Any]) -> float:
        """
        Runs a single backtest with the given strategy parameters and returns the target metric.
        (Similaire à la version de TO, mais utilise execute_backtest de trading_analyzer)
        """
        logger.info(f"Évaluation des paramètres : {current_strategy_params}")
        
        try:
            # Utiliser execute_backtest qui prend déjà les composants et strategy_params_override
            backtest_run_result = trading_analyzer.execute_backtest(
                components=self.components,
                strategy_params_override=current_strategy_params
            )
        except Exception as e:
            logger.error(f"Exception durant l'exécution du backtest avec params {current_strategy_params}: {e}", exc_info=True)
            return -float('inf') if self.maximize_metric else float('inf')

        # La structure de retour de execute_backtest contient 'metrics', 'trades', 'days', etc.
        # On s'intéresse à 'metrics'
        metrics_from_run = backtest_run_result.get('metrics', {}) if backtest_run_result else {}

        self.results.append({
            'params': current_strategy_params.copy(),
            'metrics': metrics_from_run # Stocker le dict complet des métriques
        })

        metric_value_eval = metrics_from_run.get(self.target_metric)
        if metric_value_eval is None or (isinstance(metric_value_eval, float) and np.isnan(metric_value_eval)):
            logger.warning(f"Métrique d'optimisation '{self.target_metric}' non trouvée ou NaN. Params: {current_strategy_params}")
            return -float('inf') if self.maximize_metric else float('inf')
        
        logger.info(f"Params: {current_strategy_params} -> {self.target_metric}: {metric_value_eval:.4f}")
        return float(metric_value_eval)


    def optimize(self):
        """Run the optimization process using the selected method."""
        logger.info(f"Démarrage de l'optimisation des paramètres avec la méthode : {self.optimization_method}...")
        if not self.parameter_spaces:
            logger.error("Espace des paramètres vide. Optimisation annulée.")
            return {'best_params': None, 'best_performance': None, 'all_results': [], 'runtime_seconds': 0}

        start_time = time.time()

        if self.optimization_method == 'grid':
            self._grid_search()
        elif self.optimization_method == 'random':
            self._random_search()
        elif self.optimization_method == 'bayesian':
            self._bayesian_optimization()
        else:
            logger.error(f"Méthode d'optimisation inconnue : {self.optimization_method}")
            return {'best_params': None, 'best_performance': None, 'all_results': self.results, 'runtime_seconds': time.time() - start_time}

        self._find_best_parameters() # Met à jour self.best_params, self.best_performance, self.best_metrics_package

        if self.save_results:
            self._save_results_package() # Nouvelle fonction pour sauvegarder le package complet

        end_time = time.time()
        logger.info(f"Optimisation terminée en {end_time - start_time:.2f} secondes.")
        if self.best_params:
            logger.info(f"Meilleurs paramètres trouvés : {self.best_params}")
            logger.info(f"Meilleure performance ({self.target_metric}) : {self.best_performance}")
        else:
            logger.warning("Aucun meilleur paramètre n'a été trouvé.")
        
        return {
            'best_params': self.best_params,
            'best_performance': self.best_performance,
            'best_metrics_package': self.best_metrics_package, # Ajouter le package de métriques
            'all_results': self.results,
            'runtime_seconds': end_time - start_time
        }

    def _grid_search(self):
        """Run grid search optimization."""
        logger.info("Exécution de la recherche par grille (Grid Search)...")
        
        param_combinations = self._generate_grid_combinations()
        if not param_combinations:
            logger.warning("Aucune combinaison de paramètres générée pour Grid Search.")
            return
        
        total_combinations = len(param_combinations)
        logger.info(f"{total_combinations} combinaisons de paramètres générées pour Grid Search.")
        
        current_best_score = -float('inf') if self.maximize_metric else float('inf')

        for i, params_iter in enumerate(param_combinations):
            logger.info(f"Backtest {i+1}/{total_combinations} pour Grid Search : {params_iter}")
            score = self._objective_function(params_iter)
            
            # Mettre à jour le meilleur score en direct
            if (self.maximize_metric and score > current_best_score) or \
               (not self.maximize_metric and score < current_best_score):
                current_best_score = score
            
            logger.info(f"Grid Search Progress: {i+1}/{total_combinations} | "
                       f"Score Actuel: {score:.4f} | Meilleur Score Jusqu'ici: {current_best_score:.4f}")
        # La recherche du meilleur final se fait dans _find_best_parameters après que tous les résultats soient dans self.results

    def _random_search(self):
        """Run random search optimization."""
        logger.info(f"Exécution de la recherche aléatoire (Random Search) avec {self.num_runs} itérations...")
        if not self.parameter_spaces or self.num_runs == 0:
            logger.warning("Espace de paramètres vide ou zéro itération pour Random Search.")
            return

        current_best_score = -float('inf') if self.maximize_metric else float('inf')

        for i in range(self.num_runs):
            params_iter = self._generate_random_params()
            if not params_iter: # Peut arriver si l'espace est mal défini
                logger.warning(f"Itération {i+1}: Impossible de générer des paramètres aléatoires. Saut.")
                continue

            logger.info(f"Backtest {i+1}/{self.num_runs} pour Random Search : {params_iter}")
            score = self._objective_function(params_iter)
            
            if (self.maximize_metric and score > current_best_score) or \
               (not self.maximize_metric and score < current_best_score):
                current_best_score = score

            logger.info(f"Random Search Progress: {i+1}/{self.num_runs} | "
                       f"Score Actuel: {score:.4f} | Meilleur Score Jusqu'ici: {current_best_score:.4f}")


    def _bayesian_optimization(self):
        """Run Bayesian optimization."""
        logger.info(f"Exécution de l'optimisation bayésienne avec {self.num_runs} itérations...")
        
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
        except ImportError:
            logger.error("scikit-optimize non installé. `pip install scikit-optimize`")
            return

        dimensions = []
        param_names_for_skopt = [] # Noms des paramètres passés à skopt
        
        for param, space_def in self.parameter_spaces.items():
            if space_def['type'] == 'float': # Range [min, max]
                dimensions.append(Real(space_def['range'][0], space_def['range'][1], name=param))
                param_names_for_skopt.append(param)
            elif space_def['type'] == 'int': # Range [min, max]
                dimensions.append(Integer(space_def['range'][0], space_def['range'][1], name=param))
                param_names_for_skopt.append(param)
            elif space_def['type'] in ['categorical_float', 'categorical_int', 'categorical_str']: # Liste de valeurs
                dimensions.append(Categorical(space_def['values'], name=param))
                param_names_for_skopt.append(param)
            # Les paramètres 'fixed' ne sont pas optimisés, ils seront ajoutés plus tard

        if not dimensions:
            logger.error("Aucune dimension d'optimisation définie pour l'optimisation bayésienne (tous les paramètres sont fixes?).")
            return

        # Définir la fonction objectif pour skopt
        # `@use_named_args` mappe les valeurs de `dimensions` aux arguments nommés de la fonction
        @use_named_args(dimensions=dimensions)
        def skopt_objective_function(**params_from_skopt):
            # Compléter avec les paramètres fixes
            full_params_for_backtest = params_from_skopt.copy()
            for p_name, p_space in self.parameter_spaces.items():
                if p_space['type'] == 'fixed':
                    full_params_for_backtest[p_name] = p_space['value']
            
            score = self._objective_function(full_params_for_backtest)
            # skopt minimise, donc retourner l'opposé si on maximise
            return -score if self.maximize_metric else score
        
        logger.info("Démarrage du processus d'optimisation bayésienne gp_minimize...")
        result_skopt = gp_minimize(
            func=skopt_objective_function,
            dimensions=dimensions,
            n_calls=self.num_runs,
            random_state=42, # Pour la reproductibilité
            verbose=True # Afficher la progression de skopt
        )
        
        # Les résultats sont déjà stockés dans self.results par _objective_function.
        # result_skopt.x contient les meilleurs paramètres trouvés par skopt (pour les dimensions optimisées)
        # result_skopt.fun contient la meilleure valeur de la fonction objectif (déjà ajustée pour min/max)
        logger.info(f"Optimisation bayésienne terminée. Meilleur score objectif (skopt) : {result_skopt.fun}")
        # Le _find_best_parameters final sélectionnera le meilleur globalement à partir de self.results.


    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for grid search."""
        # (Identique à PO, mais s'assure de gérer le format de self.parameter_spaces)
        param_lists_for_product = {}
        for param, space_def in self.parameter_spaces.items():
            if space_def['type'] == 'fixed':
                param_lists_for_product[param] = [space_def['value']]
            elif space_def['type'] in ['categorical_float', 'categorical_int', 'categorical_str']:
                param_lists_for_product[param] = space_def['values']
            elif space_def['type'] == 'float': # Range
                start, end = space_def['range']
                num_points = 10 # Ou configurable
                if start > 0 and end/start > 100 : # Log scale si grande plage positive
                    param_lists_for_product[param] = np.logspace(np.log10(start), np.log10(end), num_points)
                else:
                    param_lists_for_product[param] = np.linspace(start, end, num_points)
            elif space_def['type'] == 'int': # Range
                start, end = space_def['range']
                num_points = 10 # Ou configurable
                step = max(1, (end - start +1) // num_points) if (end-start+1) > 0 else 1
                param_lists_for_product[param] = list(range(start, end + 1, step))
                if not param_lists_for_product[param]: # Si step trop grand
                    param_lists_for_product[param] = [start] if start <=end else []
                # S'assurer que le dernier point est inclus si possible
                if param_lists_for_product[param] and param_lists_for_product[param][-1] < end and step > 1:
                    param_lists_for_product[param].append(end)
                param_lists_for_product[param] = sorted(list(set(param_lists_for_product[param]))) # Unique et trié


        if not param_lists_for_product: return []

        param_names = list(param_lists_for_product.keys())
        value_combinations = list(itertools.product(*[param_lists_for_product[p] for p in param_names]))
        
        return [dict(zip(param_names, combo)) for combo in value_combinations]

    def _generate_random_params(self) -> Dict[str, Any]:
        """Generate random parameters based on the defined parameter spaces."""
        # (Identique à PO, mais s'assure de gérer le format de self.parameter_spaces)
        params_rand = {}
        for param, space_def in self.parameter_spaces.items():
            if space_def['type'] == 'fixed':
                params_rand[param] = space_def['value']
            elif space_def['type'] in ['categorical_float', 'categorical_int', 'categorical_str']:
                if space_def['values']: params_rand[param] = random.choice(space_def['values'])
                else: logger.warning(f"Liste de valeurs vide pour le paramètre catégoriel '{param}'.") # Ne rien ajouter
            elif space_def['type'] == 'float': # Range
                start, end = space_def['range']
                params_rand[param] = random.uniform(start, end)
            elif space_def['type'] == 'int': # Range
                start, end = space_def['range']
                if start > end: # Correction si l'ordre est inversé
                    logger.warning(f"Range incorrect pour param int '{param}': start ({start}) > end ({end}). Inversion.")
                    start, end = end, start
                params_rand[param] = random.randint(start, end)
        return params_rand

    def _find_best_parameters(self):
        """Find the best parameters based on the target metric from self.results."""
        if not self.results:
            logger.warning("Aucun résultat pour trouver les meilleurs paramètres.")
            return

        try:
            # Default value pour la métrique si elle est absente ou NaN, pour éviter les erreurs de tri
            default_metric_val = -float('inf') if self.maximize_metric else float('inf')

            # Filtrer les résultats qui n'ont pas la métrique cible ou qui sont NaN
            valid_results = [
                r for r in self.results 
                if self.target_metric in r['metrics'] and \
                   r['metrics'][self.target_metric] is not None and \
                   not (isinstance(r['metrics'][self.target_metric], float) and np.isnan(r['metrics'][self.target_metric]))
            ]

            if not valid_results:
                logger.warning(f"Aucun résultat valide trouvé avec la métrique '{self.target_metric}'.")
                self.best_params = None
                self.best_performance = None
                self.best_metrics_package = None
                return

            # Trier les résultats valides
            sorted_results = sorted(
                valid_results,
                key=lambda x: x['metrics'][self.target_metric], # Accès direct car filtré
                reverse=self.maximize_metric
            )
            
            best_result_entry = sorted_results[0]
            self.best_params = best_result_entry['params']
            self.best_performance = float(best_result_entry['metrics'][self.target_metric])
            self.best_metrics_package = best_result_entry['metrics'] # Stocker tout le dict des métriques
            
            logger.info(f"Meilleurs paramètres trouvés : {self.best_params}")
            logger.info(f"Meilleure performance ({self.target_metric}) : {self.best_performance:.4f}")
        
        except Exception as e:
            logger.error(f"Erreur lors de la recherche des meilleurs paramètres : {e}", exc_info=True)
            self.best_params = None
            self.best_performance = None
            self.best_metrics_package = None


    def _save_results_package(self):
        """Save optimization results (params, metrics, plots, best config) to disk."""
        # (Adapté de la fonction _save_results de PO)
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir_base = self.config.get('parameter_optimizer', 'output_dir', fallback='optimization_results')
            # Créer un sous-répertoire pour cette exécution d'optimisation
            run_output_dir = os.path.join(output_dir_base, f"opt_run_{self.optimization_method}_{timestamp_str}")
            os.makedirs(run_output_dir, exist_ok=True)
            
            output_data = {
                'timestamp': timestamp_str,
                'method': self.optimization_method,
                'num_runs_or_combinations': self.num_runs if self.optimization_method != 'grid' else len(self._generate_grid_combinations()),
                'target_metric': self.target_metric,
                'maximize_metric': self.maximize_metric,
                'best_params': self.best_params,
                'best_performance': self.best_performance,
                'best_metrics_package': self.best_metrics_package,
                'parameter_spaces_used': self.parameter_spaces,
                'all_individual_results': self.results # Contient params et metrics pour chaque run
            }
            
            json_file_path = os.path.join(run_output_dir, f"full_optimization_report_{timestamp_str}.json")
            with open(json_file_path, 'w') as f:
                # Utiliser un default handler pour json.dump si des objets non sérialisables sont présents (ex: Timestamp)
                json.dump(output_data, f, indent=4, default=str) 
            logger.info(f"Rapport d'optimisation complet sauvegardé dans : {json_file_path}")
            
            if self.best_params:
                optimized_config_path = os.path.join(run_output_dir, f"optimized_strategy_params_{timestamp_str}.ini")
                self._save_optimized_config_to_file(optimized_config_path, self.best_params)
            
            # Générer les visualisations
            self._generate_and_save_visualizations(run_output_dir, timestamp_str)

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du package de résultats : {e}", exc_info=True)

    def _generate_and_save_visualizations(self, output_dir: str, timestamp_suffix: str):
        """Generate visualization plots from the optimization results and save them."""
        # (Identique à _generate_visualizations de PO, mais utilise self.results et self.target_metric)
        if not self.results: return
        try:
            results_data_list = []
            for res_item in self.results:
                # S'assurer que la métrique cible existe et est valide
                metric_val = res_item['metrics'].get(self.target_metric)
                if metric_val is None or (isinstance(metric_val, float) and np.isnan(metric_val)):
                    metric_val = np.nan # Utiliser NaN pour les graphiques si la métrique est manquante/invalide

                data_row = {'metric_value_for_plot': metric_val}
                for p_name, p_val in res_item['params'].items(): data_row[p_name] = p_val
                results_data_list.append(data_row)
            
            df_plot = pd.DataFrame(results_data_list)
            if df_plot.empty: return

            # 1. Histogramme de la métrique cible
            plt.figure(figsize=(10, 6))
            plt.hist(df_plot['metric_value_for_plot'].dropna(), bins=30, edgecolor='k')
            plt.title(f'Distribution de {self.target_metric}')
            plt.xlabel(self.target_metric)
            plt.ylabel('Fréquence')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, f"dist_{self.target_metric}_{timestamp_suffix}.png"))
            plt.close()
            
            # 2. Scatter plots pour chaque paramètre vs la métrique cible
            # Uniquement pour les paramètres qui ont été variés
            varied_params = [p for p, s_def in self.parameter_spaces.items() if s_def['type'] != 'fixed']

            for param_name_plot in varied_params:
                if param_name_plot in df_plot.columns and df_plot[param_name_plot].nunique() > 1:
                    plt.figure(figsize=(10, 6))
                    # Gérer les types catégoriels pour l'axe x si nécessaire
                    if df_plot[param_name_plot].dtype == 'object' or df_plot[param_name_plot].nunique() < 10: # Traiter comme catégoriel
                        sns.stripplot(x=param_name_plot, y='metric_value_for_plot', data=df_plot, jitter=True, alpha=0.7)
                        sns.boxplot(x=param_name_plot, y='metric_value_for_plot', data=df_plot, color='lightblue', showfliers=False)

                    else: # Numérique continu
                        plt.scatter(df_plot[param_name_plot], df_plot['metric_value_for_plot'], alpha=0.6, edgecolors='w', linewidth=0.5)
                        # Ajouter une ligne de tendance si c'est pertinent (ex: lowess)
                        try:
                            valid_data_trend = df_plot[[param_name_plot, 'metric_value_for_plot']].dropna()
                            if len(valid_data_trend) > 5 : # Assez de points pour une tendance
                                 sns.regplot(x=param_name_plot, y='metric_value_for_plot', data=valid_data_trend, scatter=False, lowess=True, line_kws={'color': 'red', 'linestyle': '--'})
                        except Exception as e_trend:
                             logger.warning(f"Impossible d'ajouter la ligne de tendance pour {param_name_plot}: {e_trend}")


                    plt.title(f'{param_name_plot} vs {self.target_metric}')
                    plt.xlabel(param_name_plot)
                    plt.ylabel(self.target_metric)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.savefig(os.path.join(output_dir, f"scatter_{param_name_plot}_vs_{self.target_metric}_{timestamp_suffix}.png"))
                    plt.close()
            
            logger.info(f"Visualisations sauvegardées dans {output_dir}")
        except Exception as e_plots:
            logger.error(f"Erreur lors de la génération des visualisations : {e_plots}", exc_info=True)

    def _save_optimized_config_to_file(self, config_filepath: str, optimized_params: Dict[str, Any]):
        """Save the optimized parameters to a new INI configuration file section."""
        # (Adapté de _save_optimized_config de PO)
        try:
            # Créer un nouveau ConfigParser ou modifier celui existant
            # Pour l'instant, créons un fichier simple avec la section [strategy_hybrid_optimized]
            
            config_out = ConfigParser()
            section_name = 'strategy_hybrid_optimized_params' # Nom de section clair
            config_out.add_section(section_name)
            
            config_out.set(section_name, f"# Optimized parameters for target_metric='{self.target_metric}' on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            config_out.set(section_name, f"# Best performance ({self.target_metric}): {self.best_performance:.4f}")

            for param_name_save, param_value_save in optimized_params.items():
                config_out.set(section_name, param_name_save, str(param_value_save))
            
            with open(config_filepath, 'w') as f_cfg_out:
                config_out.write(f_cfg_out)
            logger.info(f"Configuration optimisée sauvegardée dans : {config_filepath} (Section: {section_name})")
        
        except Exception as e_cfg_save:
            logger.error(f"Erreur lors de la sauvegarde de la configuration optimisée : {e_cfg_save}", exc_info=True)


def main_optimizer_cli():
    """Main function to run the parameter optimizer from CLI."""
    # (Similaire à la fonction main de PO)
    parser = argparse.ArgumentParser(description='Parameter Optimizer for Trading System')
    parser.add_argument('--config', default=CONFIG_FILE, help='Path to configuration file')
    parser.add_argument('--method', default='grid', choices=['grid', 'random', 'bayesian'], help='Optimization method')
    parser.add_argument('--runs', type=int, default=10, help='Number of optimization runs (for random/bayesian)') # Réduit pour tests rapides
    parser.add_argument('--target_metric', type=str, default=None, help='Metric to optimize (overrides config)')
    parser.add_argument('--minimize', action='store_true', help='Minimize the target metric (default is maximize)')
    parser.add_argument('--no-save', action='store_true', help='Do not save optimization results')
    # Pas d'argument pour --parallel pour l'instant
    
    args = parser.parse_args()
    
    optimizer_instance = ParameterOptimizer(
        config_file=args.config,
        optimization_method=args.method,
        num_runs=args.runs,
        save_results=not args.no_save,
        target_metric=args.target_metric,
        maximize_metric=not args.minimize
        # parameter_spaces_override peut être utilisé si on lance l'optimiseur depuis un autre script Python
    )
    
    optimization_summary = optimizer_instance.optimize()
    
    print("\n" + "="*60)
    print("           OPTIMIZATION SUMMARY           ")
    print("="*60)
    print(f"Method:                     {args.method}")
    print(f"Runs/Combinations:          {optimization_summary.get('num_runs_or_combinations', optimizer_instance.num_runs)}")
    print(f"Target Metric:              {optimizer_instance.target_metric}")
    print(f"Goal:                       {'Maximize' if optimizer_instance.maximize_metric else 'Minimize'}")
    print(f"Total Optimization Time:    {optimization_summary.get('runtime_seconds', 0):.2f} seconds")
    print("-" * 60)
    if optimization_summary.get('best_params'):
        print("Best Parameters Found:")
        for p_name, p_val in optimization_summary['best_params'].items():
            print(f"  {p_name:<30}: {p_val}")
        print(f"\nBest Performance ({optimizer_instance.target_metric}): {optimization_summary.get('best_performance', 'N/A'):.4f}")
        if optimization_summary.get('best_metrics_package'):
            print("\nFull Metrics for Best Run:")
            for m_name, m_val in optimization_summary['best_metrics_package'].items():
                 if isinstance(m_val, float): print(f"  {m_name:<30}: {m_val:.4f}")
                 else: print(f"  {m_name:<30}: {m_val}")
    else:
        print("No optimal parameters were found.")
    print("="*60)

if __name__ == "__main__":
    main_optimizer_cli()