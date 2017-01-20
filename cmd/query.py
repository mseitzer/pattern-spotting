#!/usr/bin/env python3
import os
import sys
import argparse

from PIL import Image

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath(os.path.split(os.path.realpath(__file__))[0]
                                + '/..'))

from src.search import SearchModel, search_roi


parser = argparse.ArgumentParser(description=
                                 'Query a database for similar images')
parser.add_argument('--database', default=None, 
                    help='Path to database to use')
parser.add_argument('--features', required=True, 
                    help='Path to features to use')
parser.add_argument('--model', required=True, 
                    help='Name or path of model to use')
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

    query_images = []
    for image_path in args.images:
        if os.path.exists(image_path):
            query_images.append(image_path)
        else:
            print('Image {} does not exist. Skipping.'.format(image_path))

    search_model = SearchModel(args.model, args.features, args.database)

    for image_path in query_images:
        image = Image.open(image_path)
        image = image.convert('RGB')

        results, similarities, bboxes = search_roi(search_model, 
                                                   image, 
                                                   roi=bbox,
                                                   top_n=args.top_n)

        print('Top {} results for query image {}'.format(len(results), 
                                                         image_path))
        for result, similarity, bbox in zip(results, similarities, bboxes):
            result_path = search_model.get_metadata(result)['image']
            print('{}\t{:.4f}\t{}'.format(result_path, similarity, bbox))

if __name__ == '__main__':
    main(sys.argv[1:])
