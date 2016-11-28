#!/usr/bin/env python3
import os
import sys
import argparse
import csv

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath(os.path.split(os.path.realpath(__file__))[0]
                                + '/..'))
from src.database import Database

parser = argparse.ArgumentParser(description=
                                 'Modify an image database')
parser.add_argument('command', choices=['CREATE', 'ADD', 'LIST'],
                    help='Action to execute')
parser.add_argument('--root-dir', dest='root_dir', default=None, 
                    help='Directory to which relatives paths are taken')
parser.add_argument('database',
                    help='Filename of the database to use')
parser.add_argument('input', nargs='?', default=None,
                    help='Either a folder or a CSV file to parse')

EXTENSIONS = ['.png', '.jpg']

def add_from_folder(db, folder, root_dir=None):
    images = []
    for root, dir_names, file_names in os.walk(folder):
        for file_name in file_names:
            ext = os.path.splitext(file_name)[1]
            if ext.lower() in EXTENSIONS:
                images.append(os.path.join(root, file_name))

    for image in images:
        if root_dir:
            image = os.path.relpath(image, root_dir)
        db.add_image(image)


def add_from_csv(db, csv_file, root_dir=None):
    with open(csv_file, 'r') as f:
        fieldnames = ['path', 'url', 'date']
        reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in reader:
            path = row['path']
            if root_dir:
                path = os.path.relpath(row['path'], root_dir)
            img = db.add_image(path)
            img['url'] = row['url']


def main(args):
    args = parser.parse_args(args)

    if args.command == 'CREATE':
        db = Database(os.path.basename(args.database))
    elif args.command in ['ADD', 'LIST']:
        db = Database.load(args.database)

    if args.command == 'LIST':
        print('Database {} contains {} images'.format(db.name, len(db)))
        for path, data in db.iter_images():
            print(path)
        return

    if not args.input:
        print('Input file/folder required')
        return

    if args.root_dir:
        args.root_dir = os.path.abspath(args.root_dir)
    args.input = os.path.abspath(args.input)

    if os.path.isdir(args.input):
        add_from_folder(db, args.input, args.root_dir)
    elif os.path.splitext(args.input)[1] == '.csv':
        add_from_csv(db, args.input, args.root_dir)
    else:
        print('Could not interpret input {}'.format(args.input))

    db.save(args.database)


if __name__ == '__main__':
    main(sys.argv[1:])

