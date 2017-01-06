import numpy as np

def _normalize(v):
    """L2 normalization of vector"""
    return v / np.linalg.norm(v, 2)


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


def _compute_mac(features):
    """
    Computes maximum activations of convolutions, which is the 
    maximum across the spatial dimensions of the features.
    """
    return np.amax(features, axis=(0,1))


def compute_r_macs(features, scales=(1, 4)):
    """
    Computes regional maximum activations of convolutions
    (see arXiv:1511.05879v2)

    scales: inclusive interval from which the scale parameter l is chosen from, 
    where l=1 is a square region filling the whole image. Higher scales result 
    in smaller regions following the formula given in the paper on p.4
    """
    assert len(features.shape) == 3
    height, width = features.shape[0], features.shape[1]

    r_macs = []
    for scale in range(scales[0], scales[1]+1):
        r_size = round(2 * min(height, width) / (scale + 1))
        if verbose:
            print('Region width at scale {}: {}'.format(scale, r_size))

        # Uniform sampling of square regions with 40% overlap
        for region in _region_generator(features, r_size, 0.4):
            r_mac = _compute_mac(features)

            # L2 normalization
            r_mac = _normalize(r_mac)

            r_macs.append(r_mac)
    return r_macs


def _compute_global_r_mac(features, pca=None):
    """
    Computes global aggregation of rmacs from convolutional features

    pca (optional): sklearn.decomposition.PCA object which is applied to each 
    mac to whiten the data
    """
    assert len(features.shape) == 3
    
    global_r_mac = np.zeros(features.shape[2])  # Sum of all regional features
    macs = compute_r_macs(features)
    
    for mac in macs:
        if pca:
            mac = pca.transform(mac)
            mac = _normalize(mac)
        global_r_mac += mac

    # Final L2 normalization
    global_r_mac = _normalize(global_r_mac)
    return global_r_mac


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
