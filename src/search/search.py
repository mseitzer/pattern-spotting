import os
import sys
import argparse

import numpy as np

from ..util import crop_image, convert_image
from ..features import compute_features, \
                       compute_representation, \
                       compute_localization_representation
from .search_model import SearchModel
from .localization import localize


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

    Returns: (indices, similarities), where indices is an array of top_n 
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
    top_n: how many query results to return. top_n=0 returns all results

    Returns: (indices, similarities, bounding_boxes), where indices is an 
        array of top_n indices of entries in the feature_store sorted 
        by decreasing similarity, similarities contains the 
        corresponding similarity score for each entry, and bounding boxes 
        is a list of tuples of the form (left, upper, right, lower) 
        specifying the rough location of the found objects.
    """
    assert top_n >= 0
    avg_query_exp_n = 5  # How many top entries to use in query expansion

    crop = convert_image(crop_image(image, roi))
    crop = search_model.preprocess_fn(crop)

    query_features = compute_features(search_model.model, crop)
    query_repr = compute_representation(query_features, search_model.pca)

    # Step 1: initial retrieval
    print('Retrieval')
    indices, _ = query(query_repr, search_model.feature_store, top_n)

    # Step 2: localization and re-ranking
    print('Localization')
    bounding_boxes = np.empty(len(indices), dtype=(int, 4))
    bounding_box_reprs = np.empty((len(indices), query_repr.shape[-1]))
    localization_repr = compute_localization_representation(query_features)
    for idx, feature_idx in enumerate(indices):
        features = search_model.get_features(feature_idx)
        bounding_boxes[idx] = localize(localization_repr, 
                                       features, crop.shape[1:])

        x1, y1, x2, y2 = bounding_boxes[idx]
        bbox_repr = compute_representation(features[x1:x2+1,y1:y2+1])
        bounding_box_reprs[idx] = bbox_repr

        # Map bounding box coordinates to image coordinates
        # TODO: maybe do this only for bounding boxes really in the final list
        metadata = search_model.get_metadata(feature_idx)
        scale_x = metadata['width'] / features.shape[1]
        scale_y = metadata['height'] / features.shape[0]
        bounding_boxes[idx][0] *= scale_x
        bounding_boxes[idx][1] *= scale_y
        bounding_boxes[idx][2] *= scale_x
        bounding_boxes[idx][3] *= scale_y
        
    reranking_indices, _ = query(query_repr, bounding_box_reprs)

    # Step 3: average query expansion
    print('Query expansion')
    best_rerank_indices = reranking_indices[:avg_query_exp_n]
    avg_repr = np.average(np.vstack((bounding_box_reprs[best_rerank_indices], 
                                     query_repr)), axis=0)
    exp_indices, similarity = query(avg_repr, bounding_box_reprs)

    # Construct bounding boxes return list
    bbox_list = []
    for bbox in bounding_boxes[exp_indices]:
        bbox_list.append((bbox.item(0), bbox.item(1), 
                          bbox.item(2), bbox.item(3)))

    return indices[exp_indices], similarity, bbox_list
