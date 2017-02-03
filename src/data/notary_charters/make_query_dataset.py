#!/usr/bin/env python3
"""Script to create query crops from labeled annotations"""
import os
import sys
import argparse
from collections import defaultdict

from PIL import Image

from annotations import (parse_labeled_annotations, write_labeled_annotations)

parser = argparse.ArgumentParser(description='Create notary charters '
                                 'query dataset')
parser.add_argument('--size', default=None, type=int, 
                    help='Maximum side size of full images')
parser.add_argument('image_dir', help='Folder where images are read from')
parser.add_argument('output_dir', help='Output directory of query crops')
parser.add_argument('annotations_file', help='Labeled annotations file')

def main(args):
    args = parser.parse_args(args)

    crops_per_label = defaultdict(list)

    for name, bbox, label in parse_labeled_annotations(args.annotations_file):
        image_path = os.path.join(args.image_dir, name)

        image = Image.open(image_path)

        scale = None
        if args.size is not None:
            if args.size < image.width or args.size < image.height:
                scale_x = args.size / image.width 
                scale_y = args.size / image.height
                scale = min(scale_x, scale_y)

        image = image.crop(bbox)

        if scale is not None:
            image = image.resize((round(scale * image.width), 
                                  round(scale * image.height)), Image.BILINEAR)
            bbox = (round(bbox[0] * scale), round(bbox[1] * scale), 
                    round(bbox[2] * scale), round(bbox[3] * scale))

        class_dir = os.path.join(args.output_dir, str(label))
        if not os.path.isdir(class_dir):
            os.mkdir(class_dir)
    
        crop_name = '{:02d}_{}'.format(len(crops_per_label[label]), name)
        crops_per_label[label].append((crop_name, bbox))
        image.save(os.path.join(class_dir, crop_name))

    labeled_crops_path = os.path.join(args.output_dir, 'labeled_crops.csv')
    labeled_crops = ((image, bbox, label) 
                     for label in sorted(crops_per_label.keys())
                     for image, bbox in crops_per_label[label])
    write_labeled_annotations(labeled_crops_path, labeled_crops)
    print('Wrote {} labeled crops '
          'to {}'.format(len(labeled_crops), labeled_crops_path))

if __name__ == '__main__':
    main(sys.argv[1:])