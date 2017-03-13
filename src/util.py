import numpy as np
import keras.preprocessing.image as keras_image

def normalize(v):
    """L2 normalization of vector"""
    norm = np.linalg.norm(v, 2)
    if norm == 0.0:
        return v
    else:
        return v / norm
    

def crop_image(image, bounding_box):
    """Crops a PIL image to a bounding box. 
    
    Performs bounding box size verification. 
    Args:
    bounding_box: Bounding box in the form of (left, upper, right+1, lower+1). 
        If bounding_box is None, no action is performed.
    """
    width, height = image.size
    if bounding_box is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        x1, y1, x2, y2 = bounding_box
    x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0,
                             [width, height, width, height])
    if x1 >= x2 or y1 >= y2:
        raise ValueError('Region of interest out of range')

    # PIL crop excludes x2/y2 coordinates from the crop
    return image.crop((x1, y1, x2, y2))


def convert_image(image):
    """Converts an RGB PIL image to Keras input format
    
    Returns: The image as a numpy array of the shape (height, width, channels)
    """
    image = keras_image.img_to_array(image)
    return image


def load_image(image_path):
    """Loads an image from disk to Keras input format

    Returns: The image as a numpy array of the shape (height, width, channels)
    """
    image = keras_image.load_img(image_path)
    return convert_image(image)
