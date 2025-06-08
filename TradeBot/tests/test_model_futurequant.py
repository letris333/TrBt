import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import os
import json 

from src.models.model_futurequant import FutureQuantModel 
from src.utils.config_manager import ConfigManager

class TestFutureQuantModel(unittest.TestCase):

    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.mock_config_manager = MagicMock(spec=ConfigManager)
        
        self.model_data_path = 'test_futurequant_model_data.json' 
        self.mock_config_manager.get_model_path.return_value = self.model_data_path 
        self.mock_config_manager.get_model_config.return_value = {
            'quantiles': [0.1, 0.5, 0.9], 
            'lookback_period': 20,       
        }

        self.dummy_model_data = {
            'trained_quantiles': {
                'q_0.1': 100.0,
                'q_0.5': 105.0,
                'q_0.9': 110.0
            },
            'some_other_parameter': 'value'
        }
        with open(self.model_data_path, 'w') as f:
            json.dump(self.dummy_model_data, f)

        self.fq_model = FutureQuantModel(config_manager=self.mock_config_manager)
        if not hasattr(self.fq_model, 'logger'):
            self.fq_model.logger = MagicMock()

        self.sample_features_df = pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10) * 100
        })

    def tearDown(self):
        """Nettoyage après chaque test."""
        if os.path.exists(self.model_data_path):
            os.remove(self.model_data_path)

    def test_initialization(self):
        """Teste l'initialisation correcte de FutureQuantModel."""
        self.assertIsNotNone(self.fq_model)
        self.assertIsNone(self.fq_model.model_data) 

    def test_load_model_data_success(self):
        """Teste le chargement réussi des données/état du modèle FutureQuant."""
        self.fq_model.load_model_data() 
        self.assertIsNotNone(self.fq_model.model_data)
        self.assertEqual(self.fq_model.model_data['trained_quantiles']['q_0.5'], 105.0)

    def test_load_model_data_file_not_found(self):
        """Teste la gestion de l'erreur si le fichier de données du modèle n'est pas trouvé."""
        if os.path.exists(self.model_data_path):
            os.remove(self.model_data_path) 
        
        with self.assertLogs(self.fq_model.logger, level='ERROR') as cm:
            self.fq_model.load_model_data()
        self.assertIsNone(self.fq_model.model_data)
        self.assertTrue(any("Failed to load FutureQuant model data" in log_message for log_message in cm.output))

    def test_predict(self):
        """Teste la méthode de prédiction de FutureQuantModel."""
        self.fq_model.load_model_data() 
        self.assertIsNotNone(self.fq_model.model_data, "Model data should be loaded before prediction")
        
        predictions = self.fq_model.predict(self.sample_features_df)
        
        self.assertIsNotNone(predictions)
        self.assertIsInstance(predictions, list) 
        self.assertEqual(len(predictions), len(self.sample_features_df))

    def test_predict_model_data_not_loaded(self):
        """Teste la prédiction si les données/état du modèle n'ont pas été chargés."""
        self.fq_model.model_data = None 
        
        with self.assertLogs(self.fq_model.logger, level='ERROR') as cm:
            predictions = self.fq_model.predict(self.sample_features_df)
        
        self.assertIsNone(predictions)
        self.assertTrue(any("FutureQuant model data not loaded" in log_message for log_message in cm.output))

if __name__ == '__main__':
    unittest.main()
