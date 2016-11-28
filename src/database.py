import pickle

class Database:
    def __init__(self, name):
        self.name = name
        self.images = {}

    def add_image(self, image_path):
        """Adds an image to the database"""
        img_dict = {
            'url': ''  # External URL linked to this image
        }
        self.images[image_path] = img_dict
        return img_dict

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
