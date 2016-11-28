import pickle

class Database:
    """Simple image database managing meta informations about images

    Members:
        name: filename of the database object
        images: contains a dict mapping filenames to a metadata dict.
                possible keys in the metadata dict:
                url: an external URL linked to this image
    """

    def __init__(self, name):
        self.name = name
        self.images = {}

    def __len__(self):
        return len(self.images)

    def add_image(self, image_path):
        """Adds an image to the database"""
        metadata_dict = {}
        self.images[image_path] = metadata_dict
        return metadata_dict

    def iter_images(self):
        return iter(self.images.items())

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
