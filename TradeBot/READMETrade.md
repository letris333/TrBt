# Système de Trading Hybride Avancé

Un système de trading automatisé intégrant l'analyse de Structure de Marché (MS), l'Order Flow (OF), la Gestion Dynamique du Risque (DRM), et des modèles ML pour le trading multi-actifs.

## Architecture du Système

Le système combine plusieurs approches pour prendre des décisions de trading robustes:

1. **Modèles de Machine Learning**:
   - XGBoost: Classification (Achat/Vente/Conservation)
   - FutureQuant: Modèle Transformer pour prédictions de quantiles de prix

2. **Analyse Technique Avancée**:
   - Market Structure: Profil de volume, POC, Value Area, HVN/LVN
   - Order Flow: CVD, Absorption, Trapped Traders
   - Pi-Ratings: Système propriétaire d'évaluation de la force haussière/baissière
   - Indicateurs TA-Lib: RSI, MACD, Bandes de Bollinger, etc.

3. **Gestion Dynamique du Risque**:
   - Classification de la qualité du setup (A/B/C)
   - Ajustement du sizing en fonction de la qualité
   - Stratégies de sortie avancées (Move-to-BE, Trailing Stop)

4. **Analyse de Sentiment**:
   - Traitement des actualités pour évaluer le sentiment du marché.
   - La colonne contenant les scores de sentiment est configurable via `config.ini` (clé `sentiment_col` dans la section `[backtest]`).

5. **Décision Multi-Facteur**:
   - Système de scoring avec confluence
   - Exigences strictes pour l'entrée en position
   - Conditions de sortie adaptatives

6. **Système de Trading Adaptatif**:
   - Conçu pour s'adapter aux conditions changeantes du marché en détectant les régimes de marché (volatilité, tendance).
   - Intègre l'analyse du carnet d'ordres (si les données sont disponibles).
   - Conditionne la validité des signaux et l'exécution des stratégies sur ces informations contextuelles, passant d'une logique `SI pattern_X ALORS strategie_X` à `SI pattern_X ET regime_Y [ET order_book_Z] ALORS probabilité_succès=P, engager_stratégie_X_avec_paramètres_adaptés`.
   - Établit des critères clairs d'invalidation des modèles et des stratégies.
   - Pour les modèles ML : favorise le réentraînement régulier et l'incorporation de caractéristiques reflétant le régime de marché et les caractéristiques du carnet d'ordres.

## Structure des Fichiers

```
├── main_trader.py           # Point d'entrée principal avec gestion d'erreurs robuste
├── config.ini               # Configuration globale du système
├── indicators.py            # Module unifié d'indicateurs (Pi-Ratings + TA)
├── feature_engineer.py      # Préparation des features pour les modèles
├── strategy_hybrid.py       # Logique de décision combinant tous les signaux
├── training_pipeline.py     # Préparation des données et entraînement des modèles
├── model_xgboost.py         # Modèle XGBoost pour classification
├── model_futurequant.py     # Modèle Transformer pour prédiction de quantiles
├── market_structure_analyzer.py # Analyse de la structure de marché (profil volume)
├── order_flow_analyzer.py   # Analyse des flux d'ordres
├── parameter_optimizer.py   # Optimisation des paramètres de trading
├── labelling.py             # Création des labels et cibles d'entraînement
├── data_handler.py          # Récupération des données de marché en temps réel
├── db_handler.py            # Gestion des interactions avec la base de données SQL
├── data_collector.py        # Collecte périodique de données financières
├── sentiment_analyzer.py    # Analyse de sentiment des actualités
├── position_manager.py      # Gestion des positions ouvertes
├── order_manager.py         # Exécution des ordres sur les exchanges avec stratégies de sortie avancées
├── tests/                   # Suite de tests unitaires et d'intégration
│   ├── test_order_flow_analyzer.py
│   ├── test_position_manager.py
│   └── ...
├── README.md                # Documentation du projet
└── requirements.txt         # Dépendances Python
```

## Composants Clés

### Features de Market Structure

L'analyse de la structure de marché utilise principalement le profil de volume pour identifier:

- **Point of Control (POC)**: Niveau de prix avec le plus grand volume
- **Value Area (VA)**: Zone contenant ~70% du volume (VA High & VA Low)
- **High Volume Nodes (HVN)**: Nœuds à fort volume, potentiels supports/résistances
- **Low Volume Nodes (LVN)**: Nœuds à faible volume, zones de vide potentielles

### Gestion Dynamique du Risque (DRM)

Le système DRM permet de:

1. **Évaluer la Qualité du Setup**:
   - Setup A: Forte confluence (70%+ du score max)
   - Setup B: Confluence moyenne (50-70%)
   - Setup C: Confluence faible (30-50%)

2. **Ajuster le Sizing**:
   - Setup A: 100% de la taille de position standard
   - Setup B: 60% de la taille standard
   - Setup C: 30% de la taille standard

### Stratégies de Sortie Avancées

Nouvellement implémentées dans `order_manager.py`:

- **Move-to-Breakeven**: Déplace automatiquement le SL au point d'entrée après un profit de X%
- **Trailing Stop**: Ajuste dynamiquement le SL à X% sous le plus haut depuis l'entrée
- **Fonctions dédiées**: `update_stop_loss()` et `update_trailing_stop()` pour gérer ces stratégies

### Gestion d'Erreurs Production

Implémentée dans `main_trader.py` avec les fonctionnalités suivantes:

- Mécanismes de retry pour les appels API avec backoff exponentiel
- Procédures d'arrêt d'urgence en cas de problèmes critiques
- Surveillance du heartbeat du système
- Vérifications périodiques de l'état du système
- Système de notification pour les défaillances critiques

### Optimisation des Paramètres

Module `parameter_optimizer.py` fournissant:

- Recherche par grille (Grid Search)
- Recherche aléatoire (Random Search)
- Optimisation bayésienne
- Visualisations des performances selon différents paramètres

### Suite de Tests

Le projet utilise `pytest` comme framework de test standard.
Tests unitaires pour valider les composants clés:
- `test_order_flow_analyzer.py`: Validation de l'analyse OF
- `test_position_manager.py`: Validation de la gestion de position

## Configuration

Le fichier `config.ini` contient tous les paramètres du système:

- Clés API pour les exchanges et fournisseurs de données
- Paramètres de trading (paires d'actifs, taille des positions, etc.)
- Configuration des modèles ML (XGBoost, FutureQuant)
- Seuils pour la structure de marché et l'order flow
- Paramètres de la gestion dynamique du risque
- Paramètres pour les stratégies de sortie avancées
- Paramètres pour l'analyse de sentiment:
  - `sentiment_col` (dans la section `[backtest]`): Spécifie le nom de la colonne utilisée pour les scores de sentiment dans les données préparées.
- Paramètres spécifiques au Système Trinary (dans la section `[trinary_config]`):
  - `trinary_rsi_period`: Période de lookback pour le RSI utilisé dans le calcul de momentum du marché (R_IFT).
  - `trinary_atr_period_stability`: Période de lookback pour l'ATR utilisé dans le calcul du facteur de stabilité (R_IFT).
  - `trinary_atr_period_volatility`: Période de lookback pour l'ATR (volatilité actuelle) utilisé dans le calcul de R_DVU.
  - `trinary_atr_sma_period`: Période de lookback pour la SMA de l'ATR de volatilité, utilisée comme ATR moyen dans le calcul de R_DVU.
  - `trinary_bbw_percentile_window`: Fenêtre de lookback pour calculer le rang percentile du Bollinger Band Width (BB_WIDTH), qui détermine `volatility_cycle_pos` pour R_DVU.

## Démarrage

1. Installer les dépendances:
```
pip install -r requirements.txt
```

2. Configurer les clés API dans `config.ini`

3. Lancer le système:
```
python main_trader.py
```

## Flow de Trading

1. **Initialisation**: Chargement des modèles, scalers et état des indicateurs
2. **Boucle Principale**:
   - Récupération des données récentes
   - Analyse de la structure de marché et de l'order flow
   - Mise à jour des Pi-Ratings et indicateurs
   - Analyse du sentiment des actualités
   - Prédictions des modèles ML
   - Évaluation de la qualité du setup (A/B/C)
   - Décision hybride combinant tous les signaux
   - Exécution des ordres selon la décision
   - Gestion des positions ouvertes avec stratégies de sortie avancées

## Formation Continue et Optimisation

- Pipeline d'entraînement pour mise à jour périodique des modèles ML
- Optimisation des paramètres via `parameter_optimizer.py`
- Backtest pour validation des stratégies via le module de backtesting

## Limitations et Considérations

- **Données Order Flow**: L'analyse OF complète nécessite des données tick/L2 non disponibles via les APIs standard
- **Backtesting**: La validation des stratégies MS/OF nécessite des données historiques détaillées
- **Adaptation aux Marchés**: Les paramètres doivent être ajustés selon les instruments et timeframes

---

**AVERTISSEMENT**: Ce système est fourni à des fins éducatives et de recherche. Le trading comporte des risques financiers significatifs. Aucune garantie de performance n'est offerte ou implicite. 