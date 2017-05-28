import os
import sys
import argparse
import threading

import numpy as np

from src.features import (compute_features, compute_representation, 
                          compute_localization_representation)
from src.search.search_model import SearchModel
from src.search.localization_jit import localize

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


def _localize_parallel(search_model, query_features, feature_idxs, image_shape,
                       n_threads=8):
    """Localizes where a query occurs on a number of features

    Args:
    search_model: instance of the SearchModel class
    query_features: features of the image to query for
    features_idxs: N indices of the features to query on
    image_shape: shape of the query image in the form of (height, width)
    n_threads: number of threads to use in parallel

    Returns: array of N bounding boxes in the form of (left, upper, 
        right, lower).
    """
    def f(res, feature_idxs):
        for idx, feature_idx in enumerate(feature_idxs):
            features = search_model.get_features(feature_idx)
            res[idx] = localize(localization_repr, features, image_shape)

    localization_repr = compute_localization_representation(query_features)
    bounding_boxes = np.empty(len(feature_idxs), dtype=(int, 4))

    threads = []
    chunk_len = int(np.ceil(len(feature_idxs) / n_threads))
    for i in range(n_threads):
        args = [bounding_boxes[i * chunk_len:(i + 1) * chunk_len],
                feature_idxs[i * chunk_len:(i + 1) * chunk_len]]
        threads.append(threading.Thread(target=f, args=args))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return bounding_boxes


def _localize(search_model, query_features, feature_idxs, image_shape):
    """Localizes where a query occurs on a number of features

    Args:
    search_model: instance of the SearchModel class
    query_features: features of the image to query for
    features_idxs: N indices of the features to query on
    image_shape: shape of the query image in the form of (height, width)

    Returns: array of N bounding boxes in the form of (left, upper, 
        right, lower).
    """
    localization_repr = compute_localization_representation(query_features)
    bounding_boxes = np.empty(len(feature_idxs), dtype=(int, 4))
    for idx, feature_idx in enumerate(feature_idxs):
        features = search_model.get_features(feature_idx)
        bounding_boxes[idx] = localize(localization_repr, features, image_shape)

    return bounding_boxes


def _compute_bbox_reprs(search_model, bounding_boxes, feature_idxs):
    """Computes representations for bounding boxes on feature maps

    Args:
    search_model: instance of the SearchModel class
    bounding_boxes: list of N bounding boxes in the form of (left, upper, 
        right, lower)
    feature_idxs: list of N indices specifying the feature maps for 
        each bounding box

    Returns: array of represesentations in the shape of (N, D), where D is 
        the representation size
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

    Query expansion works by averaging the best N representations found so far 
    with the query representation. The resulting representation vector is 
    often more distinctive for the actually searched object than the query 
    representation. Requerying the feature representations with the averaged 
    representation thus leads to improved results.

    Args:
    query_repr: representation of the search query
    feature_reprs: array of M feature representations
    feature_idxs: list of M indices specifying the order of representations
    top_n: how many representations to use in the expansion

    Returns: (indices, similarities), where indices is an array of M entries 
        specifying the order of entries in feature_reprs by decreasing 
        similarity, and similarities contains the corresponding similarity 
        scores for each entry.
    """
    reprs = np.vstack((feature_reprs[feature_idxs[:top_n]], query_repr))
    avg_repr = np.average(reprs, axis=0)
    indices, similarities = _query(avg_repr, feature_reprs)
    return indices, similarities


def _map_bboxes(search_model, bboxes, features_idxs):
    """Maps bounding boxes on feature maps to bounding boxes on images

    The mapping scales the box by the ratio between image and feature map size.
    As we have no way (yet) to deduct the feature map shape directly from an 
    image size for a specific model, we cache the feature map sizes in the 
    feature metadata.

    Args:
    search_model: instance of the SearchModel class
    bboxes: array of the shape (N, 4), where each row specifies the bounding 
        box in the form of (left, upper, right, lower)
    feature_idxs: list of N indices specifying the feature maps corresponding 
        to each bounding box

    Returns: list of N bounding boxes in the form of (left, upper, right, lower)
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


def search(search_model, query, top_n=0, localize=True, localize_n=50, 
           rerank=True, avg_qe=True):
    """Search the feature store for a query

    Args:
    search_model: instance of the SearchModel class
    query: array to search for in the shape of (height, width, 3)
    roi: bounding box in the form of (left, upper, right, lower)
    top_n: how many query results to return. top_n=0 returns all results
    localize: perform localization of objects, i.e. find bounding boxes
    localize_n: how many top images to perform localization on
    rerank: rerank images after localization by using representations 
        on the found bounding boxes
    avg_qe: perform average query expansion

    Returns: (indices, similarities, bounding_boxes), where indices is an 
        array of top_n indices of entries in the feature_store sorted 
        by decreasing similarity, similarities contains the 
        corresponding similarity score for each entry, and bounding boxes 
        is a list of tuples of the form (left, upper, right, lower) 
        specifying the rough location of the found objects if localizing, 
        None otherwise.
    """
    assert top_n >= 0
    if rerank:
        assert localize, 'Rerank implies localization'

    bboxes = None
    reprs = search_model.feature_store

    query_features = compute_features(search_model.model, query)
    query_repr = compute_representation(query_features, search_model.pca)

    retrieval_n = localize_n if localize else 0
    feature_idxs, sims = _query(query_repr, reprs, retrieval_n)
    idxs = feature_idxs

    if localize:
        bboxes = _localize_parallel(search_model, query_features, idxs, 
                                    query.shape[:2])

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
