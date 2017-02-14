#!/usr/bin/env python3
import os
import sys
import argparse
import json
from collections import OrderedDict, defaultdict

from PIL import Image

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath(os.path.split(os.path.realpath(__file__))[0] 
                                + '/..'))
from src.data.notary_charters.annotations import (parse_labeled_annotations, 
                                                  write_labeled_annotations)
from src.search import SearchModel, search_roi

parser = argparse.ArgumentParser(description='Evaluate a model\'s performance')
parser.add_argument('config', help='Search model config to use')
parser.add_argument('query_dataset', help='Path to query dataset')
parser.add_argument('predictions_file', 
                    help='File saving predictions to or containing predictions')

# Number of images a label must have to be considered as a query
MIN_RELEVANT_ELEMENTS = 5

def avg_precision(actual, predictions):
    k = min(len(actual), len(predictions))
    score = 0.0
    correct_items = 0.0
    for idx, predicted in enumerate(predictions):
        if predicted in actual:
            correct_items += 1.0
            score += (correct_items / (idx+1))
    return score / k


def run_predictions(search_model, queries, rerank_n, map_n):
    predictions = OrderedDict()
    for query in queries:
        image = Image.open(query)
        results, _, bboxes = search_roi(search_model, image, 
                                        top_n=rerank_n, 
                                        verbose=False)
        predictions[query] = []
        for result, bbox in zip(results[:map_n], bboxes[:map_n]):
            result_path = search_model.get_metadata(result)['image']
            predictions[query].append((result_path, bbox))
    return predictions


def main(args):
    args = parser.parse_args(args)

    with open(args.config, 'r') as f:
        config = json.load(f)

    crops_per_label = defaultdict(list)
    for name, bbox, label in parse_labeled_annotations(args.query_dataset):
        crop_path = os.path.join(os.path.dirname(args.query_dataset), 
                                 str(label), name)
        crops_per_label[label].append((crop_path, bbox))

    # Filter queries
    if 0 in crops_per_label: del crops_per_label[0]
    crops_per_label = {k: v for k, v in crops_per_label.items() 
                       if len(v) >= MIN_RELEVANT_ELEMENTS}

    queries = {q[0]: label for label in crops_per_label.values()
                           for q in label}
    
    if not os.path.exists(args.predictions_file):
        search_model = SearchModel.from_config(config)
        predictions = run_predictions(search_model, sorted(queries), 
                                      config['rerank_n'], config['map_n'])
        with open(args.predictions_file, 'w') as f:
            json.dump(predictions, f)
    else:
        with open(args.predictions_file, 'r') as f:
            predictions = json.load(f)

    precision_sum = 0.0
    for query, expected in queries.items():
        # Compare only filenames.
        # This leads to problem in the case of naming conflicts
        expected = [os.path.basename(l[0]).split('_', 1)[1] for l in expected]
        results = [os.path.basename(l[0]) for l in predictions[query]]
        precision_sum += avg_precision(expected, results)

    mean_avg_precision = precision_sum / len(queries)
    print('mAP@{}: {:.4f}'.format(config['map_n'], mean_avg_precision))

if __name__ == '__main__':
    main(sys.argv[1:])
