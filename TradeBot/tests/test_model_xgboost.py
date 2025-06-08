import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import xgboost as xgb 
import joblib 
import os

from src.models.model_xgboost import XGBoostModel 
from src.utils.config_manager import ConfigManager

class TestXGBoostModel(unittest.TestCase):

    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.mock_config_manager = MagicMock(spec=ConfigManager)
        
        self.model_path = 'test_xgboost_model.json' 
        self.mock_config_manager.get_model_path.return_value = self.model_path
        self.mock_config_manager.get_model_config.return_value = {
            'objective': 'multi:softprob', 
            'num_class': 3, 
            # ... autres paramètres de configuration du modèle
        }

        self.dummy_booster = xgb.XGBClassifier(objective='multi:softprob', n_estimators=1, use_label_encoder=False)
        X_dummy = np.random.rand(10, 5) 
        y_dummy = np.random.randint(0, 3, 10) 
        self.dummy_booster.fit(X_dummy, y_dummy)
        self.dummy_booster.save_model(self.model_path)

        self.xgb_model = XGBoostModel(config_manager=self.mock_config_manager)
        if not hasattr(self.xgb_model, 'logger'):
            self.xgb_model.logger = MagicMock()

        self.sample_features_df = pd.DataFrame(np.random.rand(5, 5), columns=[f'feature_{i}' for i in range(5)])

    def tearDown(self):
        """Nettoyage après chaque test."""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_initialization(self):
        """Teste l'initialisation correcte de XGBoostModel."""
        self.assertIsNotNone(self.xgb_model)
        self.assertIsNone(self.xgb_model.model) 

    def test_load_model_success(self):
        """Teste le chargement réussi du modèle XGBoost."""
        self.xgb_model.load_model()
        self.assertIsNotNone(self.xgb_model.model)
        self.assertIsInstance(self.xgb_model.model, xgb.Booster) 

    def test_load_model_file_not_found(self):
        """Teste la gestion de l'erreur si le fichier du modèle n'est pas trouvé."""
        if os.path.exists(self.model_path):
            os.remove(self.model_path) 
        
        with self.assertLogs(self.xgb_model.logger, level='ERROR') as cm:
            self.xgb_model.load_model() 
        self.assertIsNone(self.xgb_model.model)
        self.assertTrue(any("Failed to load XGBoost model" in log_message for log_message in cm.output))

    def test_predict(self):
        """Teste la méthode de prédiction."""
        self.xgb_model.load_model() 
        self.assertIsNotNone(self.xgb_model.model, "Model should be loaded before prediction")
        
        predictions = self.xgb_model.predict(self.sample_features_df)
        
        self.assertIsNotNone(predictions)
        self.assertIsInstance(predictions, np.ndarray) 
        self.assertEqual(len(predictions), len(self.sample_features_df)) 
        
        self.assertTrue(True) 

    def test_predict_model_not_loaded(self):
        """Teste la prédiction si le modèle n'a pas été chargé."""
        self.xgb_model.model = None
        
        with self.assertLogs(self.xgb_model.logger, level='ERROR') as cm:
            predictions = self.xgb_model.predict(self.sample_features_df)
        
        self.assertIsNone(predictions)
        self.assertTrue(any("Model not loaded" in log_message for log_message in cm.output))

if __name__ == '__main__':
    unittest.main()
