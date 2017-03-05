#!/usr/bin/env python3
import os
import sys
import argparse
from os.path import join, basename

from PIL import Image, ImageDraw

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath(os.path.split(os.path.realpath(__file__))[0]
                                + '/..'))
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

def main(args):
    args = parser.parse_args(args)

    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        bbox = None

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
        image = Image.open(image_path).convert('RGB')
        if bbox:
            image = crop_image(image, bbox)
        image = convert_image(image)
        
        results, similarities, bboxes = search(search_model, image, top_n=0)

        N = args.top_n
        print('Top {} results for query image {}'.format(N, image_path))
        for result, similarity, bbox in zip(results[:N], 
                                            similarities[:N], 
                                            bboxes[:N]):
            result_path = search_model.get_metadata(result)['image']

            print('{}\t{:.4f}\t{}'.format(result_path, similarity, bbox))

            if args.output:
                image = Image.open(join(args.image_dir, result_path))
                image.convert('RGB')
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, outline=(255, 0, 0))
                path = join(args.output, '{}_{:.4f}_{}'.format(
                    basename(image_path), similarity, basename(result_path)
                ))
                image.save(path)

if __name__ == '__main__':
    main(sys.argv[1:])
