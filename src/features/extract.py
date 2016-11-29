import numpy as np

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

    r_mac = _compute_r_mac(features)
    return r_mac


def representation_size(model):
    # Note that this does not work for both Tensorflow and Theano backends
    return model.layers[-1].output_shape[3]


def convert_image(image):
    """Converts an RGB PIL image to Keras input format"""
    import keras.preprocessing.image as keras_image
    image = keras_image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def load_image(image_path):
    """Loads an image to Keras input format"""
    import keras.preprocessing.image as keras_image
    image = keras_image.load_img(image_path)
    return convert_image(image)
