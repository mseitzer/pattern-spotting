#!/usr/bin/env python3
import os
import sys
import argparse
import json

import numpy as np

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath(os.path.split(os.path.realpath(__file__))[0]
                                + '/..'))
from src.models import load_model
from src.features import compute_representation, \
                         representation_size, \
                         load_image

parser = argparse.ArgumentParser(description=
                                 'Extract feature representations')
parser.add_argument('--features-dir', dest='features_dir',
                    help='Folder where extracted features are stored',
                    default='../features')
parser.add_argument('--root-dir', dest='root_dir', default=None, 
                    help='Directory to which relative paths are taken')
parser.add_argument('--image-dir', dest='image_base_dir',
                    help='Folder where images are read from', 
                    default='../data/working')
parser.add_argument('--model', default='VGG16',
                    help='Name of model or path to model definition')
parser.add_argument('name', 
                    help='(Dataset) name to use for extracted files')


def main(args):
    args = parser.parse_args(args)

    if args.root_dir:
        args.root_dir = os.path.abspath(args.root_dir)
    args.image_base_dir = os.path.abspath(args.image_base_dir)

    extensions = ['.png', '.jpg']
    images = os.listdir(args.image_base_dir)
    images = [img for img in images if os.path.splitext(img)[1] in extensions]

    model, preprocess_fn = load_model(args.model)

    meta_data = {}
    feature_store = np.empty((len(images), representation_size(model)))

    for idx, image_name in enumerate(images):
        print('{}/{}: extracting features of image {}'.format(idx+1, 
                                                              len(images), 
                                                              image_name))
        image_path = os.path.join(args.image_base_dir, image_name)
        image = load_image(image_path)
        image = preprocess_fn(image)

        features = compute_representation(model, image)
        feature_store[idx] = features
        meta_data[idx] = {
            'image': os.path.relpath(image_path, args.root_dir)
        }

    meta_file_name = '{}.meta'.format(args.name)
    with open(os.path.join(args.features_dir, meta_file_name), 'w') as f:
        json.dump(meta_data, f)

    feature_file_name = '{}.npy'.format(args.name)
    np.save(os.path.join(args.features_dir, feature_file_name), feature_store)
    

if __name__ == '__main__':
    main(sys.argv[1:])
