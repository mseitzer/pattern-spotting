#!/usr/bin/env python3
import os
import sys
import argparse
import json
from collections import defaultdict
from timeit import default_timer as timer

from PIL import Image

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath(os.path.split(os.path.realpath(__file__))[0] 
                                + '/..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.data.notary_charters.annotations import (parse_labeled_annotations, 
                                                  write_labeled_annotations)
from src.util import convert_image
from src.search import SearchModel, search

parser = argparse.ArgumentParser(description='Benchmark query speed of a model')
parser.add_argument('config', help='Search model config to use')
parser.add_argument('query_dataset', help='Path to query dataset')

# Number of images a label must have to be considered as a query
MIN_RELEVANT_ELEMENTS = 2

def warmup_jit(search_model, queries, rerank_n, map_n):
    image = convert_image(Image.open(queries[0])) 
    search(search_model, image, top_n=map_n, localize_n=rerank_n)


def time_predictions(search_model, queries, rerank_n, map_n):
    total_time = 0
    
    for idx, query in enumerate(queries):
        print('Running query {}/{}: {}'.format(idx+1, len(queries), query))
        image = convert_image(Image.open(query)) 
       
        start_time = timer()
        search(search_model, image, top_n=map_n, localize_n=rerank_n)
        end_time = timer()
        total_time += end_time - start_time
        
    return total_time


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
    
    search_model = SearchModel.from_config(config)
    
    for i in range(10):
        warmup_jit(search_model, sorted(queries), config['rerank_n'], map_n)

    total_time = time_predictions(search_model, sorted(queries),
                                  config['rerank_n'], map_n)

    print('Average time for {} queries: {:.4f} seconds'.format(len(queries), 
        total_time / len(queries)))


if __name__ == '__main__':
    main(sys.argv[1:])
