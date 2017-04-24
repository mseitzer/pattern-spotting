#!/usr/bin/env python3
"""Script to create a binary classification dataset charter/no charter"""
import os
import sys
import argparse

import numpy as np
from PIL import Image

from .annotations import parse_annotations

parser = argparse.ArgumentParser(description='Create binary classification '
                                 'dataset charter/no charter')
parser.add_argument('--size', default=256, type=int, 
                    help='Size of resulting images')
parser.add_argument('--val-frac', default=0.2, type=float, 
                    help='Percentage of images to use for validation')
parser.add_argument('image_dir', help='Folder where images are read from')
parser.add_argument('annotations_dir',
                    help='Folder where notary charters annotations are stored')
parser.add_argument('output_dir', help='Output directory of dataset')


SCALES = [1.0, 0.75, 0.5]

def crop_image(image, bbox, scale, crop_size):
    """Crop out a centered object scaled to a fraction of the crop size

    Args:
    image: PIL image where object appears on
    bbox: bounding box in the form of (left, upper, right, lower)
    scale: fraction of crop that the object fills
    crop_size: size in pixels of the crop 

    Returns: cropped PIL image
    """
    assert crop_size <= 1.0
    # @Todo: implement
    pass


def main(args):
    args = parser.parse_args(args)

    annotation_files = sorted(os.listdir(args.annotations_dir))
    annotation_files = [os.path.join(args.annotations_dir, p) 
                        for p in annotation_files
                        if os.path.splitext(p)[1].lower() == '.xml']
    print('Found {} annotation files in {}'.format(len(annotation_files),
                                                   args.annotations_dir))

    for name, bbox in parse_annotations(annotation_files, 'GraphicRegion'):
        image_path = os.path.join(args.image_dir, name)
        if not os.path.isfile(image_path):
            # Try if file with uc/lc extension exists
            alt_name, ext = os.path.splitext(name)
            alt_name = '{}{}'.format(alt_name, ext.swapcase())
            image_path = os.path.join(image_dir, alt_name)
            if not os.path.isfile(image_path):
                print('Warning: image {} does not exist'.format(name))
                continue

        image = Image.open(image_path)
        # @Todo: implement


if __name__ == '__main__':
    main(sys.argv[1:])