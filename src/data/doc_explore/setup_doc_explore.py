#!/usr/bin/env python3
"""Script to setup the DocExplore challenge dataset

The URLs to the dataset can be retrieved after registering at 
http://spotting.univ-rouen.fr/?page_id=9
"""

import os
import sys
import argparse
import subprocess
import zipfile
import shutil
from glob import glob
from urllib.parse import urlparse

from PIL import Image

parser = argparse.ArgumentParser(description='Setup DocExplore dataset')
parser.add_argument('-o', '--output', dest='target_dir',
                    help='Folder to output dataset to', default='.')
parser.add_argument('images_zip', help='URL/Path to DocExplore_images.zip')
parser.add_argument('queries_zip', help='URL/Path to DocExplore_queries_web.zip')

IMAGES_MD5 = '4cd1520bb9587ce2350100af55410909'
QUERIES_MD5 = '9cff1795581e210645f5936b272cb24d'

def is_url(url):
    return urlparse(url).scheme != ""


def filename_from_url(url):
    return os.path.basename(urlparse(url).path)


def maybe_download_and_verify(url, ref_md5):
    if is_url(url):
        path = os.path.join('/tmp', filename_from_url(url))
        subprocess.call(['wget', '--no-clobber', '-P', '/tmp', url])
    else:
        path = url

    md5 = subprocess.check_output(['md5sum', path])
    md5 = md5.decode('utf-8').split(' ')[0]

    if md5 != ref_md5:
        raise ValueError('MD5 of {} does not match {}'.format(path, ref_md5))

    return path


def extract_zip(file, target_dir, lvl=0):
    with zipfile.ZipFile(file, 'r') as zf:
        for file in zf.namelist():
            if not os.path.basename(file):  # Is not directory
                continue
            name = os.path.join(*((os.path.normpath(file).split('/'))[lvl:]))
            sub_dir = os.path.join(target_dir, os.path.dirname(name))
            if not os.path.isdir(sub_dir):
                os.makedirs(sub_dir)
            source = zf.open(file)
            target = open(os.path.join(target_dir, name), "wb")
            with source, target:
                shutil.copyfileobj(source, target)


def main(args):
    args = parser.parse_args(args)

    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)

    images_path = maybe_download_and_verify(args.images_zip, IMAGES_MD5)
    queries_path = maybe_download_and_verify(args.queries_zip, QUERIES_MD5)
    
    extract_zip(images_path, args.target_dir, lvl=1)
    extract_zip(queries_path, os.path.join(args.target_dir, 'query'), lvl=1)

    for file in glob('{}/*.jpg'.format(args.target_dir)):
        with Image.open(file) as img:
            size = img.size
            if size[0] > 1024 or size[1] > 1024:
                img.thumbnail((1024, 1024))
                print('Image {} has size {}. '
                      'Resizing to {}'.format(file, size, img.size))
                img.save(file)


if __name__ == '__main__':
    main(sys.argv[1:])