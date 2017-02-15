import os
import sys
import argparse

import numpy as np

from src.util import crop_image, convert_image
from src.features import (compute_features, compute_representation, 
                          compute_localization_representation)
from src.search.search_model import SearchModel
from src.search.localization import localize

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


def _query(query_features, feature_store, top_n=0):
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


def _localize(search_model, query_features, feature_idxs, image_shape):
    """Localizes where a query occurs on a number of features

    Args:
    search_model: instance of the SearchModel class
    query_features: features of the image to query for
    features_idxs: N indices of the features to query on
    image_shape: shape of the original image in the form of (height, width)

    Returns: array of N bounding boxes in the form of 
        (left, upper, right, lower).
    """
    localization_repr = compute_localization_representation(query_features)
    bounding_boxes = np.empty(len(feature_idxs), dtype=(int, 4))
    for idx, feature_idx in enumerate(feature_idxs):
        features = search_model.get_features(feature_idx)
        bounding_boxes[idx] = localize(localization_repr, features, image_shape)

    return bounding_boxes


def _compute_bbox_reprs(search_model, bounding_boxes, feature_idxs):
    """@Todo:

    @Todo: explain
    @Todo: docs
    """
    repr_size = search_model.feature_store.shape[-1]
    bounding_box_reprs = np.empty((len(feature_idxs), repr_size))
    for idx, bbox in enumerate(bounding_boxes):
        x1, y1, x2, y2 = bbox
        features = search_model.get_features(feature_idxs[idx])
        bbox_repr = compute_representation(features[y1:y2+1, x1:x2+1],
                                           search_model.pca)
        bounding_box_reprs[idx] = bbox_repr
    return bounding_box_reprs


def _average_query_exp(query_repr, feature_reprs, feature_idxs, top_n=5):
    """Performs average query expansion

    @Todo: explain
    @Todo: docs
    """
    reprs = np.vstack((feature_reprs[feature_idxs[:top_n]], query_repr))
    avg_repr = np.average(reprs, axis=0)
    indices, similarity = _query(avg_repr, feature_reprs)
    return indices, similarity


def _map_bboxes(search_model, bboxes, features_idxs):
    """@Todo: docs
    """
    mapped_bboxes = []
    for bbox, feature_idx in zip(bboxes, features_idxs):
        metadata = search_model.get_metadata(feature_idx)
        scale_x = metadata['width'] / float(metadata['feature_width'])
        scale_y = metadata['height'] / float(metadata['feature_height'])
        mapped_bboxes.append((round(bbox.item(0)*scale_x), 
                              round(bbox.item(1)*scale_y), 
                              round(bbox.item(2)*scale_x), 
                              round(bbox.item(3)*scale_y)))
    return mapped_bboxes


def search_roi_new(search_model, image, roi=None, top_n=0, localize_n=50,
                   localize=True, rerank=True, avg_qe=True):
    """Query the feature store for a region of interest on an image
    @Todo: Update docs

    Args:
    search_model: instance of the SearchModel class
    image: RGB PIL image to take roi of
    roi: bounding box in the form of (left, upper, right, lower)
    top_n: how many query results to return. top_n=0 returns all results

    Returns: (indices, similarities, bounding_boxes), where indices is an 
        array of top_n indices of entries in the feature_store sorted 
        by decreasing similarity, similarities contains the 
        corresponding similarity score for each entry, and bounding boxes 
        is a list of tuples of the form (left, upper, right, lower) 
        specifying the rough location of the found objects.
    """
    assert top_n >= 0
    if rerank:
        assert localize, 'Rerank implies localization'

    bboxes = None
    reprs = search_model.feature_store

    # @Todo: remove possibility of cropping, directly pass in numpy array
    crop = convert_image(crop_image(image, roi))
    query_features = compute_features(search_model.model, crop)
    query_repr = compute_representation(query_features, search_model.pca)

    retrieval_n = localize_n if localize else top_n
    feature_idxs, sims = _query(query_repr, reprs, retrieval_n)
    idxs = feature_idxs

    if localize:
        bboxes = _localize(search_model, query_features, idxs, crop.shape[:2])

    if rerank:
        reprs = _compute_bbox_reprs(search_model, bboxes, idxs)
        idxs, sims = _query(query_repr, reprs)

    if avg_qe:
        idxs, sims = _average_query_exp(query_repr, reprs, idxs)

    if top_n > 0:
        idxs = idxs[:top_n]

    if localize:
        bboxes = _map_bboxes(search_model, bboxes[idxs], feature_idxs[idxs])

    return feature_idxs[idxs], sims, bboxes


def search_roi(search_model, image, roi=None, top_n=0, verbose=True):
    """Query the feature store for a region of interest on an image

    Args:
    search_model: instance of the SearchModel class
    image: RGB PIL image to take roi of
    roi: bounding box in the form of (left, upper, right, lower)
    top_n: how many query results to return. top_n=0 returns all results

    Returns: (indices, similarities, bounding_boxes), where indices is an 
        array of top_n indices of entries in the feature_store sorted 
        by decreasing similarity, similarities contains the 
        corresponding similarity score for each entry, and bounding boxes 
        is a list of tuples of the form (left, upper, right, lower) 
        specifying the rough location of the found objects.
    """
    assert top_n >= 0
    AVG_QUERY_EXP_N = 5  # How many top entries to use in query expansion

    crop = convert_image(crop_image(image, roi))

    query_features = compute_features(search_model.model, crop)
    query_repr = compute_representation(query_features, search_model.pca)
    localization_repr = compute_localization_representation(query_features)

    scale_x = crop.shape[1] / query_features.shape[1]
    scale_y = crop.shape[0] / query_features.shape[0]

    if verbose:
        from timeit import default_timer as timer
        start = timer()

    # Step 1: initial retrieval
    indices, sims = _query(query_repr, search_model.feature_store, top_n)

    if verbose:
        end = timer()
        print('Retrieval took {:.6f} seconds'.format(end-start))
        start = timer()

    # Step 2: localization and re-ranking
    feature_shapes = {}
    bounding_boxes = np.empty(len(indices), dtype=(int, 4))
    bounding_box_reprs = np.empty((len(indices), query_repr.shape[-1]))
    
    for idx, feature_idx in enumerate(indices):
        features = search_model.get_features(feature_idx)
        feature_shapes[feature_idx] = features.shape

        bounding_boxes[idx] = localize(localization_repr, features, 
                                       crop.shape[:2])

        x1, y1, x2, y2 = bounding_boxes[idx]
        bbox_repr = compute_representation(features[y1:y2+1, x1:x2+1], 
                                           search_model.pca)
        bounding_box_reprs[idx] = bbox_repr
        
    reranking_indices, sims = _query(query_repr, bounding_box_reprs)

    if verbose:
        end = timer()
        print('Localization took {:.6f} seconds'.format(end-start))
        start = timer()

    # Step 3: average query expansion
    best_rerank_indices = reranking_indices[:AVG_QUERY_EXP_N]
    avg_repr = np.average(np.vstack((bounding_box_reprs[best_rerank_indices], 
                                     query_repr)), axis=0)
    exp_indices, similarity = _query(avg_repr, bounding_box_reprs)

    if verbose:
        end = timer()
        print('Average query expansion took {:.6f} seconds'.format(end-start))

    # Construct bounding boxes return list
    bbox_list = []
    for feature_idx, bbox in zip(indices[exp_indices], 
                                 bounding_boxes[exp_indices]):
        # Map bounding box coordinates to image coordinates
        metadata = search_model.get_metadata(feature_idx)
        scale_x = metadata['width'] / feature_shapes[feature_idx][1]
        scale_y = metadata['height'] / feature_shapes[feature_idx][0]
        bbox_list.append((round(bbox.item(0)*scale_x), 
                          round(bbox.item(1)*scale_y), 
                          round(bbox.item(2)*scale_x), 
                          round(bbox.item(3)*scale_y)))

    return indices[exp_indices], similarity, bbox_list
