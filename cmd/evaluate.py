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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.data.notary_charters.annotations import (parse_labeled_annotations, 
                                                  write_labeled_annotations)
from src.util import convert_image
from src.search import SearchModel, search

parser = argparse.ArgumentParser(description='Evaluate a model\'s performance')
parser.add_argument('config', help='Search model config to use')
parser.add_argument('query_dataset', help='Path to query dataset')
parser.add_argument('predictions_file', 
                    help='File saving predictions to or containing predictions')

# Number of images a label must have to be considered as a query
MIN_RELEVANT_ELEMENTS = 5
# Minimum IoU a localized object must reach to be considered a hit
MIN_LOCALIZATION_IOU = 0.5

def avg_precision(expected, predictions):
    k = min(len(expected), len(predictions))
    score = 0.0
    correct_items = 0.0
    for idx, predicted in enumerate(predictions):
        if predicted in expected:
            correct_items += 1.0
            score += (correct_items / (idx+1))
    return score / k


def intersection_over_union(expected, exp_bboxes, predictions, pred_bboxes):
    def iou(bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
        intersect_area = max(0, (x2 - x1 + 1)) * max(0, (y2 - y1 + 1))
        union_area = area1 + area2 - intersect_area
        return intersect_area / union_area

    correct_items = 0
    ious = []
    for i, expect in enumerate(expected):
        correct_prediction = False
        for j, predicted in enumerate(predictions):
            if predicted == expect:
                correct_items += 1
                ious.append(iou(exp_bboxes[i], pred_bboxes[j]))
                correct_prediction = True
                break
        if not correct_prediction:
            ious.append(0.0)
    return ious, correct_items


def run_predictions(search_model, queries, rerank_n, map_n):
    predictions = OrderedDict()
    for idx, query in enumerate(queries):
        print('Running query {}/{}: {}'.format(idx+1, len(queries), query))
        image = convert_image(Image.open(query)) 
        results, _, bboxes = search(search_model, image, 
                                    top_n=map_n, 
                                    localize_n=rerank_n)
        predictions[query] = []
        for result, bbox in zip(results, bboxes):
            result_path = search_model.get_metadata(result)['image']
            predictions[query].append((result_path, bbox))
    return predictions


def main(args):
    args = parser.parse_args(args)

    with open(args.config, 'r') as f:
        config = json.load(f)

    map_n = config['map_n']

    crops_per_label = defaultdict(list)
    for name, bbox, label in parse_labeled_annotations(args.query_dataset):
        crop_path = os.path.join(os.path.dirname(args.query_dataset), 
                                 str(label), name)
        crops_per_label[label].append((crop_path, bbox))

    # Filter queries
    if '0' in crops_per_label: 
        del crops_per_label['0']
    crops_per_label = {k: v for k, v in crops_per_label.items() 
                       if len(v) >= MIN_RELEVANT_ELEMENTS}

    queries = {c[0]: crops for crops in crops_per_label.values()
                           for c in crops}
    
    # Get predictions
    predictions = None
    if os.path.exists(args.predictions_file):
        with open(args.predictions_file, 'r') as f:
            predictions = json.load(f)
        for query, preds in predictions.items():
            if len(preds) < map_n:
                print('Not enough predictions for mAP@{}, '
                      'rerunning queries'.format(map_n))
                predictions = None
                break
            predictions[query] = preds[:map_n]

    if predictions is None:
        search_model = SearchModel.from_config(config)
        predictions = run_predictions(search_model, sorted(queries), 
                                      config['rerank_n'], map_n)
        with open(args.predictions_file, 'w') as f:
            json.dump(predictions, f)

    # Evaluate predictions
    retrieval_precision_sum = 0.0
    localization_precision_sum = 0.0
    iou_sum = 0.0
    correct_items = 0
    total_items = 0

    for query, expected in queries.items():
        # Compare only filenames.
        # This leads to problem in the case of naming conflicts
        exp_items = [os.path.basename(l[0]).split('_', 1)[1] for l in expected]
        retr_items = [os.path.basename(l[0]) for l in predictions[query]]
        retrieval_precision_sum += avg_precision(exp_items, retr_items)

        exp_bboxes = [l[1] for l in expected]
        retr_bboxes = [l[1] for l in predictions[query]]

        ious, n_correct = intersection_over_union(exp_items, exp_bboxes, 
                                                  retr_items, retr_bboxes)
        iou_sum += sum(ious)
        correct_items += n_correct
        total_items += len(ious)
 
        for iou, exp in zip(ious, exp_items):
            if iou < MIN_LOCALIZATION_IOU:
                if exp in retr_items:
                    # Remove all badly localized items
                    retr_items[retr_items.index(exp)] = ''
        localization_precision_sum += avg_precision(exp_items, retr_items)

    retrieval_map = retrieval_precision_sum / len(queries)
    avg_iou = iou_sum / correct_items
    localization_map = localization_precision_sum / len(queries)
    
    print('Retrieval performance:')
    print('\tmAP@{}: {:.4f}'.format(map_n, retrieval_map))
    print('Localization performance:')
    print('\tmAP@{}: {:.4f}, at >={} IoU'.format(map_n, localization_map, 
                                                 MIN_LOCALIZATION_IOU))
    print('\tIoU over {}/{} correctly retrieved '
          'images: {:.4f}'.format(correct_items, total_items, avg_iou))

if __name__ == '__main__':
    main(sys.argv[1:])
