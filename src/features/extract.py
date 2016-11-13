import os, sys
import argparse
import numpy as np

from . import database
from models import load

parser = argparse.ArgumentParser(description=
                                 'Extract feature representations')
parser.add_argument('--database-dir', dest='database_dir',
                    help='Folder where results are stored',
                    default='../database')
parser.add_argument('--image-dir', dest='image_base_dir',
                    help='Folder where images are read from', 
                    default='../data/working')

def _region_generator(array, size, overlap, verbose=False):
    """A generator which returns square overlapping regions"""
    height, width = array.shape[0], array.shape[1]
    row = 0
    while row < height:
        row_end = min(row + size, height)

        col = 0
        while col < width:
            col_end = min(col + size, width)
            if verbose:
                print('Current region: row {}:{}, col {}:{}'.format(
                    row, row_end, col, col_end))

            yield array[row:row_end, col:col_end]

            if col + size == width:
                break
            col = round(col + (1 - overlap) * (col + size))

        if row + size == height:
            break
        row = round(row + (1 - overlap) * (row + size))


def _compute_r_mac(features, verbose=False):
    """
    Computes regional maximum activations of convolutions
    (see arXiv:1511.05879)
    """
    def normalize(v):
        return v / np.linalg.norm(v, 2)

    assert len(features.shape) == 3

    min_scale = 1
    max_scale = 4
    height, width = features.shape[0], features.shape[1]

    r_mac = np.zeros(features.shape[2])  # Sum of all regional features

    for scale in range(min_scale, max_scale+1):
        r_size = round(2 * min(height, width) / (scale + 1))
        if verbose:
            print('Region width at scale {}: {}'.format(scale, r_size))

        # Uniform sampling of square regions with 40% overlap
        for region in _region_generator(features, r_size, 0.4):
            max_region_activations = np.amax(region, axis=(0,1))

            # L2 normalization
            max_region_activations = normalize(max_region_activations)

            # TODO: PCA whitening, see sklearn.decomposition.PCA 
            # TODO: another L2 normalization

            r_mac += max_region_activations

    # Final L2 normalization
    r_mac = normalize(r_mac)

    return r_mac


def compute_representation(model, image):
    """
    Computes a representation of the image suitable to retrieval
    """
    features = model.predict(image)
    features = np.squeeze(features, axis=0)
    #features = np.random.rand(44, 62, 512)

    r_mac = _compute_r_mac(features)
    return r_mac


def representation_size(model):
    return model.layers[-1].output_shape[3]


def load_image(image_path):
    """Loads an image into a representation suitable to input into Keras models
    """
    import keras.preprocessing.image as keras_image
    image = keras_image.load_img(image_path)
    image = keras_image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def main(args):
    args = parser.parse_args(args)

    extensions = ['.png', '.jpg']
    images = os.listdir(args.image_base_dir)
    images = [img for img in images if os.path.splitext(img)[1] in extensions]

    model_name = 'VGG16'  # TODO: make argument

    db = database.Database(model_name)

    model, preprocess_fn = load(model_name)

    feature_store = np.empty((len(images), representation_size(model)))

    for idx, image_name in enumerate(images):
        image_path = os.path.join(args.image_base_dir, image_name)
        image = load_image(image_path)
        image = preprocess_fn(image)

        features = compute_representation(model, image)
        feature_store[idx] = features
        db.add_image(image_name, idx)

    database.save(db, os.path.join(args.database_dir, 'working.pkl'))
    np.save(os.path.join(args.database_dir, 'working.npy'), feature_store)

if __name__ == '__main__':
    # Note: run from src/ with python3 -m features.extract
    main(sys.argv[1:])
