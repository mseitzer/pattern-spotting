#!/usr/bin/env python3
import os
import sys
import argparse
from timeit import default_timer as timer
from os.path import join, basename

from PIL import Image, ImageDraw

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath(os.path.split(os.path.realpath(__file__))[0]
                                + '/..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.util import convert_image, crop_image
from src.search import SearchModel, search

parser = argparse.ArgumentParser(description=
                                 'Query a database for similar images')
parser.add_argument('--database', default=None, 
                    help='Path to database to use')
parser.add_argument('--features', required=True, 
                    help='Path to features to use')
parser.add_argument('--model', required=True, 
                    help='Name or path of model to use')
parser.add_argument('--output', 
                    help='Folder to output the found images '
                    'with bounding box annotations')
parser.add_argument('--image-dir',
                    help='Image directory. Required with the output option.')
parser.add_argument('-n', dest='top_n', type=int, default=5, 
                    help='How many search results to display')
parser.add_argument('--bbox', type=int, nargs=4, default=None,
                    metavar=('left', 'upper', 'right', 'lower'),
                    help='Specify a bounding box on the query')
parser.add_argument('images', nargs='+', 
                    help='One or more images to query for')


def draw_bbox_and_save(image_path, target_path, bbox):
    with Image.open(image_path) as image:
        image.convert('RGB')
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox, outline=(255, 0, 0))
        image.save(target_path)


def main(args):
    args = parser.parse_args(args)

    if args.bbox:
        args.bbox = tuple(args.bbox)
    
    if args.output and not args.image_dir:
        print('Image directory required with option output')
        return

    query_images = []
    for image_path in args.images:
        if os.path.exists(image_path):
            query_images.append(image_path)
        else:
            print('Image {} does not exist. Skipping.'.format(image_path))

    search_model = SearchModel(args.model, args.features, args.database)

    for image_path in query_images:
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            if args.bbox:
                image = crop_image(image, args.bbox)
            query = convert_image(image)
        
        start_time = timer()
        results, similarities, bboxes = search(search_model, query, 
                                               top_n=args.top_n)
        end_time = timer()

        print('Search took {:.6f} seconds'.format(end_time-start_time))
        print('Top {} results for query image {}'.format(args.top_n, 
                                                         image_path))
        for result, similarity, bbox in zip(results, similarities, bboxes):
            result_path = search_model.get_metadata(result)['image']

            print('{}\t{:.4f}\t{}'.format(result_path, similarity, bbox))

            if args.output:
                result_path = join(args.image_dir, result_path)
                target_path = join(args.output, '{}_{:.4f}_{}'.format(
                    basename(query_path), similarity, basename(result_path)
                ))
                draw_bbox_and_save(result_path, target_path, bbox)
                

if __name__ == '__main__':
    main(sys.argv[1:])
