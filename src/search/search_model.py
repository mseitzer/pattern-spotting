import json
from os.path import basename, join, isfile

import numpy as np
from sklearn.externals import joblib

from src.database import Database
from src.models import load_model
from src.features import representation_size

class SearchModel:
    """Encapsulates all components necessary to search on a database"""
    @staticmethod
    def from_config(config):
        """Constructs a search model from a config dictionary"""
        if 'model' not in config or 'features' not in config:
            raise ValueError('Search model needs model and features parameters')
        database = config.get('database')
        return SearchModel(config['model'], config['features'], database)

    def __init__(self, model, features_path, database_path=None):
        # Load the extraction model
        self.model = load_model(model)

        # Load the feature metadata
        features_basename = basename(features_path)
        meta_file_path = join(features_path, 
                              '{}.meta'.format(features_basename))
        with open(meta_file_path, 'r') as f:
            self.feature_metadata = json.load(f)
        
        # Load image representations
        repr_file_path = join(features_path, 
                              '{}.repr.npy'.format(features_basename))
        self.feature_store = np.load(repr_file_path)

        if representation_size(self.model) != self.feature_store.shape[-1]:
            raise ValueError('Model {} and feature store {} have nonmatching '
                             'representation sizes: {} vs {}'.format(
                                model, features_path,
                                representation_size(self.model),
                                self.feature_store.shape[-1]))

        # Construct paths to feature files
        self.feature_file_paths = {} 
        features_sub_folder = join(features_path, 'features/')
        for idx, metadata in self.feature_metadata.items():
            if not idx.isdigit():
                continue
            image_name = basename(self.feature_metadata[str(idx)]['image'])
            path = join(features_sub_folder, '{}.npy'.format(image_name))
            if isfile(path):
                self.feature_file_paths[str(idx)] = path
            else:
                print('Missing feature file for image {}'.format(image_name))

        # Load PCA
        pca_file_path = join(features_path, '{}.pca'.format(features_basename))
        if isfile(pca_file_path):
            self.pca = joblib.load(pca_file_path)
        else:
            self.pca = None

        # Load image database
        if database_path:
            self.database = Database.load(database_path)
        else:
            self.database = None
        
    def get_metadata(self, feature_idx):
        return self.feature_metadata[str(feature_idx)]

    def get_features(self, feature_idx):
        feature_file_path = self.feature_file_paths[str(feature_idx)]
        return np.load(feature_file_path)

    def query_database(self, image):
        if self.database:
            return self.database.images.get(image) 
        return None

