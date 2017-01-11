import os
import json

import numpy as np
from sklearn.externals import joblib

from ..database import Database
from ..models import load_model
from ..features import representation_size

class SearchModel:
    """Encapsulates all components necessary to search on a database"""
    def __init__(self, model, features_path, database_path=None):
        self.model, self.preprocess_fn = load_model(model)

        # Avoid Keras lazy predict function construction
        self.model._make_predict_function()

        features_basename = os.path.basename(features_path)
        meta_file_path = os.path.join(features_path,
                                      '{}.meta'.format(features_basename))
        with open(meta_file_path, 'r') as f:
            self.feature_metadata = json.load(f)
        
        repr_file_path = os.path.join(features_path,
                                      '{}.repr.npy'.format(features_basename))
        self.feature_store = np.load(repr_file_path)

        if representation_size(self.model) != self.feature_store.shape[-1]:
            raise ValueError('Model {} and feature store {} have nonmatching '
                             'representation sizes: {} vs {}'.format(
                                model, features_path,
                                representation_size(self.model),
                                self.feature_store.shape[-1]))

        pca_file_path = os.path.join(features_path,
                                     '{}.pca'.format(features_basename))
        if os.path.isfile(pca_file_path):
            self.pca = joblib.load(pca_file_path)
        else:
            self.pca = None

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
