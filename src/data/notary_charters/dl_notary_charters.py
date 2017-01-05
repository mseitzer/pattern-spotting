#!/usr/bin/env python3
"""Script to download images from monasterium.net from xml databases"""

import sys, os
import argparse
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='download images from monasterium')
parser.add_argument('-o', '--output', dest='target_folder',
                    help='Folder to save images in', default='.')
parser.add_argument('--only-index', action='store_true',
                    dest='only_index', default=False,
                    help='Only generate index file, do not download images')
parser.add_argument('input_files', nargs='+', help='Input XML files')

def parse_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    charters = []
    for charter in root.iter('charter'):
        attribs = {}
        for child in charter:
            attribs[child.tag] = child.text
        charters.append(attribs)
    return charters


def download_images(folder, images):
    for image in images:
        subprocess.call(['wget', '--no-clobber', '-P', folder, image])


def construct_index(f, folder, charters):
    for charter in charters:
        url = urllib.parse.urlparse(charter['imageFile'])
        image_path = '{}/{}'.format(folder, os.path.basename(url.path))
        f.write('{}/{};{};{}\n'.format(image_path, 
                                       charter['imageFile'],
                                       charter['url'], 
                                       charter['date']))


def main(arguments):
    args = parser.parse_args(arguments)
    
    base_folder = os.path.normpath(args.target_folder)
    if not os.path.exists(base_folder):
        print('Target folder {} does not exist'.format(base_folder))
        return

    for file in args.input_files:
        charters = parse_xml(file)

        image_urls = [charter['imageFile'] for charter in charters]

        folder_name = os.path.splitext(os.path.basename(file))[0]
        folder_path = '{}/{}'.format(base_folder, folder_name)

        csv_file = '{}/{}.csv'.format(base_folder, folder_name)
        with open(csv_file, 'w') as f:
            construct_index(f, folder_name, charters)

        if not args.only_index:
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            download_images(folder_path, image_urls)


if __name__ == '__main__':
    main(sys.argv[1:])