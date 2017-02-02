#!/usr/bin/env python3
import os
import sys
import argparse
from collections import namedtuple, OrderedDict, defaultdict

import matplotlib.pyplot as plt
from PIL import Image

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath(os.path.split(os.path.realpath(__file__))[0]
                                + '/..'))
from src.data.notary_charters.annotations import parse

parser = argparse.ArgumentParser(description='Interactively label '
                                 'charter annotations')
parser.add_argument('--overwrite', action='store_true',
                    help='Do not recover saved annotations file, start new one')
parser.add_argument('--show-stored', action='store_true', 
                    help='Show annotations which already have labels')
parser.add_argument('--only-default', action='store_true', 
                    help='Show only annotations which have the default label')
parser.add_argument('--summarize', action='store_true', 
                    help='Print a summary of the labeled annotations')
parser.add_argument('--image-dir',
                    help='Folder where images are read from')
parser.add_argument('annotations_dir',
                    help='Folder where notary charters annotations are stored')
parser.add_argument('output_file', 
                    help='Output file with labeled annotations')

ImageBbox = namedtuple('ImageBbox', 'image bbox')

def format_bbox(bbox):
    return '{} {} {} {}'.format(bbox[0], bbox[1], bbox[2], bbox[3])


def query_user(out_dir, image_dir, annotation_files, labeled_annotations, 
               show_stored=False, only_default=False):
    def prompt(imagebbox, max_label, existing_label=None):
        s = '\nImage: {}, bounding box: {}\n'.format(imagebbox.image, 
                                                     imagebbox.bbox)
        if existing_label is not None:
            s += 'Current label is {}. Choices are:\n'.format(existing_label)
        else:
            s += 'Choose a label for this bounding box. Choices are:\n'
        if max_label > 1:
            s += '[1-{l}]: existing labels from 1-{l}\n'.format(l=max_label)
        elif max_label == 1:
            s += '[1]: existing label 1\n'
        s += '[n]: new label {l}\n'.format(l=max_label+1)
        s += '[d]: default label 0\n'
        s += '[s]: skip this annotation\n'
        s += '[q]: save and quit\n'
        choice = input(s + '>')
        return choice

    def interact(imagebbox, image, existing_label=None):
        nonlocal max_label
        while True: 
            choice = prompt(imagebbox, max_label, existing_label)
            if choice == 's':
                break
            elif choice == 'q':
                return False
            elif choice == 'n':
                label = max_label+1
            elif choice == 'd':
                label = 0
            else:
                try:
                    label = int(choice)
                except ValueError:
                    print('{} is no valid choice'.format(choice))
                    continue

            if 0 <= label <= max_label+1:
                labeled_annotations[imagebbox] = label
                repr_path = os.path.join(out_dir, 
                                         'class_{}.jpg'.format(label))
                if label != 0 and not os.path.isfile(repr_path):
                    image.save(repr_path)
                    print('Using {} as representative '
                          'for class {}'.format(imagebbox.image, label))
                if label == max_label+1:
                    max_label += 1
                break
            else:
                print('{} is no valid choice'.format(choice))
        return True

    max_label = 0
    for label in labeled_annotations.values():
        max_label = max(max_label, label)

    window_active = False
    for name, bbox in parse(annotation_files, 'GraphicRegion'):
        imagebbox = ImageBbox(image=name, bbox=format_bbox(bbox))

        image_path = os.path.join(image_dir, name)
        if not os.path.isfile(image_path):
            # Try if file with uc/lc extension exists
            alt_name, ext = os.path.splitext(name)
            alt_name = '{}{}'.format(alt_name, ext.swapcase())
            image_path = os.path.join(image_dir, alt_name)
            if not os.path.isfile(image_path):
                print('Warning: image {} does not exist'.format(name))
                continue
            imagebbox = ImageBbox(image=alt_name, bbox=imagebbox.bbox)

        label = labeled_annotations.get(imagebbox, None)
        if label is not None:
            if not show_stored:
                continue
            if only_default and label != 0:
                continue
        else:
            if only_default:
                continue

        image = Image.open(image_path)
        image = image.crop(bbox)
        imgplot = plt.imshow(image)
        imgplot.figure.canvas.set_window_title(imagebbox.image)
        plt.draw()
        
        if not window_active:
            plt.show(block=False)
            window_active = True

        resume = interact(imagebbox, image, label)
        image.close()
        if not resume:
            break
            
    plt.close()


def print_summary(labeled_annotations):
    label_counts = defaultdict(int)
    for label in labeled_annotations.values():
        label_counts[label] += 1

    for label in sorted(label_counts.keys()):
        print('{}: {}'.format(label, label_counts[label]))


def main(args):
    args = parser.parse_args(args)

    out_dir = os.path.split(args.output_file)[0]

    annotation_files = sorted(os.listdir(args.annotations_dir))
    annotation_files = [os.path.join(args.annotations_dir, p) 
                        for p in annotation_files
                        if os.path.splitext(p)[1].lower() == '.xml']
    print('Found {} annotation files in {}'.format(len(annotation_files),
                                                   args.annotations_dir))

    labeled_annotations = OrderedDict()

    # Recover labeled annotations if the file already exists
    if not args.overwrite and os.path.isfile(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                name, bbox, label = line.split(';')
                imagebbox = ImageBbox(image=name, bbox=bbox)
                labeled_annotations[imagebbox] = int(label)
        print('Recovered {} previous labeled annotations '
              'from {}'.format(len(labeled_annotations), args.output_file))
    else:
        print('Opening new labeled annotations file {}'.format(args.output_file))

    if args.summarize:
        print_summary(labeled_annotations)
        return
    else:
        if not args.image_dir:
            print('Need image directory to continue')
            return

    query_user(out_dir, args.image_dir, annotation_files, labeled_annotations, 
               args.show_stored, args.only_default)

    with open(args.output_file, 'w') as f:
        for imagebbox in sorted(labeled_annotations.keys()):
            label = labeled_annotations[imagebbox]
            f.write('{};{};{}\n'.format(imagebbox.image, imagebbox.bbox, label))
        print('Saved {} labeled annotations '
              'to {}'.format(len(labeled_annotations), args.output_file))

if __name__ == '__main__':
    main(sys.argv[1:])
