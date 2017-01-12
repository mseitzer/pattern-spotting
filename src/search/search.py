import os
import sys
import argparse

import numpy as np

from ..util import crop_image, convert_image
from ..features import compute_features, compute_representation
from .search_model import SearchModel


def _descending_argsort(array, k):
    """Return indices that index the highest k values in an array"""
    indices = np.argpartition(array, -k)[-k:]
    indices = indices[np.argsort(array[indices])]  # Sort indices
    return indices[::-1]


def _ascending_argsort(array, k):
    """Return indices that index the lowest k values in an array"""
    indices = np.argpartition(array, k-1)[:k]
    indices = indices[np.argsort(array[indices])]  # Sort indices
    return indices


def query(query_features, feature_store, top_n=0):
    """Query stored features for similarity against a passed feature

    Args:
        query_features: Feature to query for, shape (1, dim)
        feature_store: 2D-array of n features to compare to, shape (n, dim)
        top_n:  if zero, return all results in descending order
                if positive, return only the best top_n results
                if negative, return only the worst top_n results

    Returns:
        (indices, similarities), where indices is an array of top_n 
        indices of entries in the feature_store sorted by decreasing 
        similarity, and similarities contains the corresponding 
        similarity value for each entry.
    """
    # Cosine similarity measure
    similarity = feature_store.dot(query_features.T).flatten()

    k = min(len(feature_store), abs(top_n))
    if top_n >= 0:  # Best top_n features
        indices = _descending_argsort(similarity, k)
    else:  # Worst top_n features
        indices = _ascending_argsort(similarity, k)
    
    return indices, similarity[indices]


def search_roi(search_model, image, roi=None, top_n=0):
    """Query the feature store for a region of interest on an image

    Args:
        image: RGB PIL image to take roi of
        roi: bounding box in the form of (left, upper, right, lower)
        feature_store: features to search on
        top_n: how many query results to return (see query())

    Returns:
        (indices, similarities), where indices is an array of top_n 
        indices of entries in the feature_store sorted by decreasing 
        similarity, and similarities contains the corresponding 
        similarity value for each entry.
    """
    crop = convert_image(crop_image(image, roi))
    crop = search_model.preprocess_fn(crop)

    features = compute_features(search_model.model, crop)
    representation = compute_representation(features, search_model.pca)

    return query(representation, search_model.feature_store, top_n)
