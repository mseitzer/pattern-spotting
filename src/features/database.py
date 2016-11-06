import pickle

class Database:
    def __init__(self, model_name):
        self.model_name = model_name
        self.images = []

    def add_image(self, image_path, idx):
        self.images.append({
            'path': image_path,  # Relative path to the image
            'feature_idx': idx  # Index referencing into the feature array
        })


def save(database, path):
    with open(path, 'wb') as f:
        pickle.dump(database, f, pickle.HIGHEST_PROTOCOL)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
