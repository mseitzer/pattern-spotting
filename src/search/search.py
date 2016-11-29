import os
import sys
import argparse

import numpy as np

from ..features import compute_representation, convert_image, load_image
from .search_model import SearchModel


def query(query_features, feature_store, top_n=0):
    """Return a list of indices of the most similar features"""
    query_features = np.expand_dims(query_features, axis=0)

    # Cosine similarity measure
    similarity = feature_store.dot(query_features.T).flatten()

    if top_n == 0:
        k = -len(similarity)+1  # Every feature
    elif top_n > 0:
        k = -top_n+1  # Only best top_n
    else:
        k = -top_n  # Only worst top_n
    
    # TODO: return only top_n results
    indices = np.argpartition(similarity, k)
    indices = indices[np.argsort(similarity[indices])]  # Sort indices

    top_similarities = similarity[indices]
    return indices, top_similarities


def search_roi(search_model, image, roi=None, top_n=0):
    """Query the feature store for a region of interest on an image

    Args:
        image: RGB PIL image to take roi of
        roi: bounding box in the form of (left, upper, right, lower)
        feature_store: features to search on
        top_n: how many query results to return

    Returns:
        (indices, similarities), where indices is a list of top_n 
        indices of entries in the feature_store sorted by decreasing 
        similarity, and similarities contains the corresponding 
        similarity value for each entry.
    """
    height, width = image.size
    if roi is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        x1, y1, x2, y2 = roi
    x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0,
                             [width, height, width, height])
    if x1 == x2 or y1 == y2:
        raise ValueError('Region of interest out of range')

    crop = image.crop((x1, y1, x2, y2))
    crop = convert_image(crop)
    crop = search_model.preprocess_fn(crop)

    features = compute_representation(search_model.model, crop)

    return query(features, search_model.feature_store, top_n)
