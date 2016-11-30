import os
import json

import numpy as np

from ..database import Database
from ..models import load_model
from ..features import representation_size

class SearchModel:
    """Encapsulates all components necessary to search on a database"""
    def __init__(self, model, features_path, database_path=None):
        self.model, self.preprocess_fn = load_model(model)

        # Avoid Keras lazy predict function construction
        self.model._make_predict_function()

        features_basename, _ = os.path.splitext(features_path)
        with open('{}.meta'.format(features_basename), 'r') as f:
            self.feature_metadata = json.load(f)
        
        self.feature_store = np.load('{}.npy'.format(features_basename))

        if representation_size(self.model) != self.feature_store.shape[-1]:
            raise ValueError('Model {} and feature store {} have nonmatching '
                             'representation sizes: {} vs {}'.format(
                                model, features_path,
                                representation_size(self.model),
                                self.feature_store.shape[-1]))

        if database_path:
            self.database = Database.load(database_path)
        else:
            self.database = None
        
    def get_metadata(self, feature_idx):
        return self.feature_metadata[str(feature_idx)]

    def query_database(self, image):
        if self.database:
            return self.database.images.get(image) 
        return None
