#!/usr/bin/env python3
"""Script to resize the notary charters to a usable size"""

import argparse
import os
import sys
from PIL import Image

parser = argparse.ArgumentParser(description='Resize notary charters images')
parser.add_argument('--data-dir', required=True, 
                    help='Data directory of raw notary charters images')
parser.add_argument('--out-dir', required=True, help='Output directory')
parser.add_argument('--size', default=1000, type=int, 
                    help='Maximum side size of images')

def main(args):
    args = parser.parse_args(args)

    extensions = ['.png', '.jpg', '.jpeg']
    images = os.listdir(args.data_dir)
    images = [img for img in images 
              if os.path.splitext(img)[1].lower() in extensions]
    images = sorted(images)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    for image in images:
        out_path = os.path.join(args.out_dir, image)
        img = Image.open(os.path.join(args.data_dir, image))
        img.thumbnail((args.size, args.size))
        img.save(out_path)

    print('Resized {} images to maximum side size {}'.format(len(images), 
                                                             args.size))

if __name__ == '__main__':
    main(sys.argv[1:])
