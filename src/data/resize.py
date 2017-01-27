#!/usr/bin/env python3
"""Script to resize images to a usable size"""

import argparse
import os
import sys
from PIL import Image

parser = argparse.ArgumentParser(description='Resize images in a folder')
parser.add_argument('--size', default=1000, type=int, 
                    help='Maximum side size of images')
parser.add_argument('input_dir', help='Input directory of images')
parser.add_argument('output_dir', help='Output directory of images')

def main(args):
    args = parser.parse_args(args)
    
    extensions = ['.png', '.jpg', '.jpeg']
    images = os.listdir(args.input_dir)
    images = [img for img in images 
              if os.path.splitext(img)[1].lower() in extensions]
    images = sorted(images)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for image in images:
        img = Image.open(os.path.join(args.input_dir, image))
        img.thumbnail((args.size, args.size))
        out_path = os.path.join(args.output_dir, image)
        img.save(out_path)

    print('Resized {} images to maximum side size {}'.format(len(images), 
                                                             args.size))

if __name__ == '__main__':
    main(sys.argv[1:])
