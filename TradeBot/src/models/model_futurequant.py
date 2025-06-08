# model_futurequant.py
import tensorflow as tf
# Import direct de keras au lieu de tensorflow.keras (plus fiable)
import keras
from keras import layers
# Utiliser keras_core pour les fonctionnalités avancées (incluant ce qui était dans tensorflow-addons)
import keras_core
import numpy as np
import pandas as pd
import logging
import os
import json # Pour sauvegarder params FQ
from typing import Optional, List, Dict, Tuple
from configparser import ConfigParser, NoSectionError, NoOptionError
from src.pipeline import feature_engineer # Pour le scaler

logger = logging.getLogger(__name__)

# --- Gestion du Modèle FutureQuant ---
_fq_model: Optional[keras.Model] = None
_fq_model_file: Optional[str] = None
_fq_quantiles: Optional[List[float]] = None # Les quantiles que ce modèle est entraîné à prédire


def initialize(config: ConfigParser):
    """Wrapper function for initialize_futurequant_model to maintain compatibility with trading_analyzer.py."""
    initialize_futurequant_model(config)

def initialize_futurequant_model(config: ConfigParser):
    """Initialise le chemin du fichier et charge le modèle/params si existants."""
    global _fq_model_file, _fq_quantiles
    _fq_model_file = config.get('futurequant', 'model_file', fallback='fq_model.h5')
    # Les quantiles sont critiques pour la prédiction, les charger de la config ou d'un fichier de params d'entraînement
    # Pour l'instant, on les charge de la config pour la prédiction, mais ils devraient être stockés avec le modèle entraîné.
    try:
        quantiles_list_str = config.get('futurequant', 'quantiles', fallback='0.1, 0.5, 0.9')
        _fq_quantiles = [float(q.strip()) for q in quantiles_list_str.split(',')]
        if not all(0 <= q <= 1 for q in _fq_quantiles):
             logger.error(f"Quantiles configurés invalides: {_fq_quantiles}. Doivent être entre 0 et 1.")
             _fq_quantiles = None
    except (ValueError, NoSectionError, NoOptionError) as e:
        logger.error(f"Erreur de configuration des quantiles FQ: {e}. Quantiles non définis.")
        _fq_quantiles = None


    load_futurequant_model() # Tenter de charger le modèle

    # Si le modèle est chargé mais pas les quantiles, invalider le modèle? Ou utiliser les quantiles de la config?
    # Utilisons les quantiles de la config pour la prédiction s'ils sont définis, sinon le modèle chargé est inutilisable.
    if _fq_model and not _fq_quantiles:
         logger.warning("Modèle FutureQuant chargé mais quantiles non définis dans la config ou fichier de params. Le modèle ne pourra pas être utilisé pour la prédiction.")
         _fq_model = None # Invalider le modèle si les cibles ne sont pas claires

    logger.info(f"FutureQuant model initialized. Model file: {_fq_model_file}. Quantiles: {_fq_quantiles}")


def load_futurequant_model():
    """Charge le modèle FutureQuant depuis le fichier."""
    global _fq_model
    _fq_model = None
    if _fq_model_file and os.path.exists(_fq_model_file):
        try:
            # Utiliser notre implémentation personnalisée de Pinball Loss
            # pour charger correctement le modèle avec sa fonction de perte
            custom_objects = {
                # Pattern pour matcher le nom générique des fonctions quantile_loss_*
                f'quantile_loss_tf_{"_".join(str(q).replace(".", "p") for q in _fq_quantiles if _fq_quantiles)}': 
                quantile_loss(_fq_quantiles) if _fq_quantiles else None
            }
            
            _fq_model = keras.models.load_model(_fq_model_file, custom_objects=custom_objects, compile=False)
            logger.info(f"FutureQuant model chargé avec succès depuis: {_fq_model_file}")
            # Recompiler si nécessaire avec la bonne fonction de perte
            if _fq_quantiles and _fq_model is not None:
                _fq_model.compile(
                    optimizer=keras.optimizers.Adam(),
                    loss=quantile_loss(_fq_quantiles)
                )
                logger.info("Modèle recompilé avec la fonction de perte Quantile Loss")

        except Exception as e:
            logger.warning(f"Impossible de charger le modèle FutureQuant depuis '{_fq_model_file}': {e}")
            _fq_model = None


def save_futurequant_model(model: keras.Model, config: ConfigParser):
    """Sauvegarde le modèle FutureQuant entraîné."""
    if model and _fq_model_file:
        try:
            dirname = os.path.dirname(_fq_model_file)
            if dirname and not os.path.exists(dirname):
                 os.makedirs(dirname)
            model.save(_fq_model_file) # Saves architecture, weights, training config
            logger.info(f"Modèle FutureQuant sauvegardé sous: {_fq_model_file}")

            # Sauvegarder les quantiles utilisés pour l'entraînement pour la cohérence future
            # Ceci devrait idéalement être fait dans training_pipeline
            try:
                 quantiles_list_str = config.get('futurequant', 'quantiles', fallback='0.1, 0.5, 0.9')
                 quantiles_list = [float(q.strip()) for q in quantiles_list_str.split(',')]
                 # Assuming training_params_file is the place for global training info
                 training_params_file = config.get('training', 'training_params_file', fallback='training_params.json')
                 if os.path.exists(training_params_file):
                      with open(training_params_file, 'r') as f:
                           params = json.load(f)
                 else:
                      params = {}
                 params['futurequant_quantiles'] = quantiles_list
                 with open(training_params_file, 'w') as f:
                      json.dump(params, f, indent=4)
                 logger.debug(f"Quantiles FQ sauvegardés dans {training_params_file}.")
            except Exception as e:
                 logger.error(f"Erreur lors de la sauvegarde des quantiles FQ: {e}")


        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle FutureQuant dans {_fq_model_file}: {e}", exc_info=True)
    elif not model:
         logger.warning("Tentative de sauvegarde d'un modèle FutureQuant None.")
    elif not _fq_model_file:
         logger.error("Nom de fichier non fourni pour la sauvegarde du modèle FutureQuant.")


# --- Définition de la Quantile Loss ---
# Implémentation manuelle de la PinballLoss (Quantile Loss) avec tensorflow directement

def quantile_loss(quantiles: List[float]):
    """
    Crée une fonction de perte Quantile Loss (Pinball Loss) pour un ensemble de quantiles,
    implémentée manuellement avec tensorflow.
    
    Args:
        quantiles: Liste de quantiles cibles (ex: [0.1, 0.5, 0.9]).
    Returns:
        Une fonction de perte Keras.
    """
    
    def pinball_loss(tau):
        """Implémentation manuelle de Pinball Loss pour un quantile spécifique tau."""
        def loss_fn(y_true, y_pred):
            error = y_true - y_pred
            return tf.reduce_mean(
                tf.maximum(tau * error, (tau - 1) * error),
                axis=-1
            )
        return loss_fn
    
    losses = [pinball_loss(q) for q in quantiles]
    
    def loss(y_true, y_pred):
        # y_true shape: (batch_size, 1) - the actual future ratio
        # y_pred shape: (batch_size, num_quantiles) - the predicted quantile values
        
        total_loss = 0
        for i, pinball_loss_fn in enumerate(losses):
            # Calculer la perte pour chaque quantile
            # Extraire la colonne correspondante de y_pred
            y_pred_q = y_pred[:, i:i+1]  # Garder la dimension
            q_loss = pinball_loss_fn(y_true, y_pred_q)
            total_loss += q_loss
        
        # Moyenne des pertes sur tous les quantiles
        return total_loss / len(losses)

    # Give the loss function a name so it can be saved/loaded
    loss.__name__ = f'quantile_loss_tf_{"_".join(str(q).replace(".", "p") for q in quantiles)}'
    return loss


# --- Définition de l'Architecture du Modèle FutureQuant ---

def build_futurequant_model(input_shape: Tuple[int, int], config: ConfigParser) -> keras.Model:
    """
    Construit le modèle FutureQuant Transformer basé sur la configuration.
    Args:
        input_shape: Tuple (window_size_in, num_features_per_step).
        config: ConfigParser object.
    Returns:
        Un modèle Keras compilé.
    """
    try:
        num_transformer_blocks = config.getint('futurequant', 'num_transformer_blocks', fallback=4)
        attention_heads = config.getint('futurequant', 'attention_heads', fallback=4)
        ff_dim = config.getint('futurequant', 'ff_dim', fallback=32)
        conv_filters = config.getint('futurequant', 'conv_filters', fallback=32)
        conv_kernel_size = config.getint('futurequant', 'conv_kernel_size', fallback=3)
        dropout_rate = config.getfloat('futurequant', 'dropout_rate', fallback=0.1)
        quantiles_list_str = config.get('futurequant', 'quantiles', fallback='0.1, 0.5, 0.9')
        quantiles_list = [float(q.strip()) for q in quantiles_list_str.split(',')]
        num_quantiles = len(quantiles_list)
        learning_rate = config.getfloat('futurequant', 'learning_rate', fallback=0.001)


        if not all(0 <= q <= 1 for q in quantiles_list):
             logger.error(f"Quantiles configurés invalides: {quantiles_list}. Doivent être entre 0 et 1. Construction modèle FQ annulée.")
             return None

    except (ValueError, TypeError, NoSectionError, NoOptionError) as e:
        logger.error(f"Erreur de configuration FutureQuant: {e}. Construction modèle FQ annulée.")
        return None

    # --- Architecture Keras ---
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Layer Normalization initiale
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    logger.debug("FQ: Initial LayerNormalization added.")

    # Empilement de blocs Transformer
    for i in range(num_transformer_blocks):
        logger.debug(f"FQ: Adding Transformer Block {i+1}/{num_transformer_blocks}")
        # Residual connection start
        residual = x

        # Multi-Head Attention part
        # Normalize input before Attention as per common practices (Post-Norm in diagram?)
        x = layers.LayerNormalization(epsilon=1e-6)(x) # Pre-Norm is also common
        attn_output = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=input_shape[-1])(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        # Add residual connection and normalize
        x = layers.Add()([residual, attn_output]) # Add Residual
        # x = layers.LayerNormalization(epsilon=1e-6)(x) # Normalize after Add (Post-Norm) - or before Add? Diagram implies before. Let's follow diagram flow + Add&Norm.
        # The diagram shows LayerNorm, then Attention, then Add Residual, then LayerNorm before Conv.
        # Let's structure as: Norm->Att->Add; Norm->Conv->Add.

        # Reset residual for the Conv part
        residual = x

        # Convolutional (Feedforward) part
        x = layers.LayerNormalization(epsilon=1e-6)(x) # Normalize before Conv
        x = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation="relu", padding="same")(x) # Padding="same" pour garder la longueur de séquence
        x = layers.Dropout(dropout_rate)(x)
        # Add residual connection and normalize
        x = layers.Add()([residual, x]) # Add Residual
        # x = layers.LayerNormalization(epsilon=1e-6)(x) # Normalize after Add (Post-Norm)

    # Global Average Pooling après les blocs Transformer
    x = layers.GlobalAveragePooling1D()(x)
    logger.debug("FQ: GlobalAveragePooling1D added.")

    # Couches Dense (MLP Head)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    # Add another Dense layer as in the text
    x = layers.Dense(ff_dim // 2, activation="relu")(x) # Assuming a decreasing size
    x = layers.Dropout(dropout_rate)(x)
    logger.debug(f"FQ: Two Dense layers ({ff_dim}, {ff_dim // 2}) added.")

    # Output layer - predict one value for each quantile
    outputs = layers.Dense(num_quantiles, activation="linear")(x)
    logger.debug(f"FQ: Output layer added ({num_quantiles} quantiles).")

    # Créer et compiler le modèle
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compiler le modèle avec la fonction de perte quantile
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=quantile_loss(quantiles_list))

    logger.info("FutureQuant model built and compiled.")
    model.summary(print_fn=logger.info) # Log model summary

    return model


def train_futurequant_model(X_train_seq: np.ndarray, y_train_ratio: np.ndarray, config: ConfigParser) -> Optional[keras.Model]:
    """Entraîne le modèle FutureQuant Transformer."""
    if X_train_seq is None or y_train_ratio is None or X_train_seq.shape[0] == 0 or X_train_seq.shape[0] != y_train_ratio.shape[0]:
        logger.error("Données d'entraînement (X_train_seq ou y_train_ratio) vides ou de tailles incohérentes. Entraînement FQ annulé.")
        return None

    try:
        epochs = config.getint('futurequant', 'epochs', fallback=100)
        batch_size = config.getint('futurequant', 'batch_size', fallback=64)
        validation_split = config.getfloat('training', 'validation_split', fallback=0.2)
        quantiles_list_str = config.get('futurequant', 'quantiles', fallback='0.1, 0.5, 0.9')
        quantiles_list = [float(q.strip()) for q in quantiles_list_str.split(',')]


    except (ValueError, TypeError, NoSectionError, NoOptionError) as e:
        logger.error(f"Erreur de configuration FutureQuant entraînement: {e}. Entraînement annulé.")
        return None

    # Remodeler y_train_ratio pour correspondre aux sorties du modèle [samples, num_quantiles]
    # Chaque cible de quantile est le même ratio réel future_price_ratio pour chaque quantile prédit.
    # La perte s'applique différemment à chaque sortie basée sur le quantile associé.
    y_train_multi_target = np.repeat(y_train_ratio, len(quantiles_list), axis=1) # Shape [samples, num_quantiles]


    logger.info(f"Début de l'entraînement FutureQuant avec {X_train_seq.shape[0]} échantillons, forme de séquence {X_train_seq.shape[1:]}.")
    logger.debug(f"Cibles d'entraînement (ratios) y_train_ratio forme: {y_train_ratio.shape}")
    logger.debug(f"Cibles multi-sorties y_train_multi_target forme: {y_train_multi_target.shape}")
    logger.debug(f"Époques: {epochs}, Batch Size: {batch_size}, Validation Split: {validation_split}")


    # Construire le modèle
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = build_futurequant_model(input_shape, config)

    if model is None:
        logger.error("Échec de la construction du modèle FutureQuant.")
        return None

    try:
        # Entraîner le modèle
        # Utiliser validation_split pour évaluer pendant l'entraînement
        history = model.fit(
            X_train_seq,
            y_train_multi_target, # Utiliser les cibles multi-sorties
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1, # Afficher la progression
            shuffle=True, # Mélanger les données d'entraînement
            # Ajouter Callbacks si désiré (ex: EarlyStopping, ModelCheckpoint)
        )
        logger.info("Entraînement du modèle FutureQuant terminé avec succès.")
        # Optionnel: Log validation loss
        # val_loss = history.history['val_loss'][-1]
        # logger.info(f"FutureQuant Validation Loss finale: {val_loss:.4f}")

        return model

    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement FutureQuant: {e}", exc_info=True)
        # tensorflow might handle ctrl+c, but ensure clean exit if possible
        return None


def predict_futurequant(X_live_seq: np.ndarray) -> Optional[np.ndarray]:
    """
    Fait une prédiction des quantiles futurs en utilisant le modèle FutureQuant chargé.
    X_live_seq doit avoir la forme [1, window_size_in, num_features].
    Retourne un array numpy [predicted_q1, predicted_q2, ..., predicted_qn] (ratios) ou None.
    """
    if _fq_model is None or _fq_quantiles is None:
        logger.error("Modèle FutureQuant ou quantiles non chargés.")
        return None
    # Vérifier la forme de l'entrée live
    expected_input_shape = (1, _fq_model.input_shape[1], _fq_model.input_shape[2])
    if X_live_seq is None or X_live_seq.shape != expected_input_shape:
        logger.error(f"Forme d'entrée live incorrecte pour FQ. Attendu {expected_input_shape}, obtenu {X_live_seq.shape if X_live_seq is not None else None}.")
        return None

    logger.debug(f"Prédiction FutureQuant pour 1 échantillon avec forme {X_live_seq.shape}...")

    try:
        # Utiliser le modèle chargé pour la prédiction
        predictions_ratios = _fq_model.predict(X_live_seq) # Shape [1, num_quantiles]
        logger.debug(f"Prédiction FQ brute (ratios): {predictions_ratios[0].tolist()}")

        # Les prédictions sont des ratios (Price(t+W)/Price(t)).
        # La stratégie hybride aura besoin de ces ratios ou des prix absolus prédits.
        # On peut convertir en prix absolus ici si on a le prix actuel, mais la stratégie peut le faire aussi.
        # Retournons les ratios prédits.

        return predictions_ratios[0] # Retourne array [ratio_q1, ratio_q2, ...]

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction FutureQuant: {e}", exc_info=True)
        return None

def get_model():
    """
    Retourne le modèle FutureQuant chargé
    """
    return _fq_model 