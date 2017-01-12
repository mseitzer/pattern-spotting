#!/usr/bin/env python3
import argparse
import json
import os
import sys
from os.path import join, exists

import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath(os.path.split(os.path.realpath(__file__))[0]
                                + '/..'))

from src.util import load_image
from src.models import load_model
from src.features import compute_features, \
                         compute_r_macs, \
                         compute_representation

parser = argparse.ArgumentParser(description=
                                 'Extract feature representations')
parser.add_argument('--features-dir',
                    help='Folder where extracted data is stored',
                    default='../features')
parser.add_argument('--root-dir', default=None, 
                    help='Directory to which relative paths are taken')
parser.add_argument('--image-dir',
                    help='Folder where images are read from', 
                    default='../data/working')
parser.add_argument('--model', default='VGG16',
                    help='Name of model or path to model definition')
parser.add_argument('command', choices=['features', 'pca', 'repr'],
                    help='Action to execute')
parser.add_argument('name', 
                    help='(Dataset) name to use for extracted files')


def extract_conv_features(name, model, features_dir, image_dir, root_dir):
    """Extracts features of all images in image_dir and 
    saves them for later use.
    """
    image_dir = os.path.abspath(image_dir)

    out_dir = join(features_dir, 'features/')
    if exists(out_dir):
        os.mkdir(out_dir)

    extensions = ['.png', '.jpg']
    images = os.listdir(image_dir)
    images = [img for img in images if os.path.splitext(img)[1] in extensions]
    images = sorted(images)

    model, preprocess_fn = load_model(model)

    meta_data = {}
    for idx, image_name in enumerate(images):
        print('{}/{}: extracting features of image {}'.format(idx+1, 
                                                              len(images), 
                                                              image_name))
        image_path = join(image_dir, image_name)
        image = load_image(image_path)
        image = preprocess_fn(image)

        features = compute_features(model, image)

        np.save(join(out_dir, os.path.basename(image_name)), features)
        meta_data[idx] = {
            'image': os.path.relpath(image_path, root_dir)
        }

    meta_file_name = '{}.meta'.format(name)
    with open(join(features_dir, meta_file_name), 'w') as f:
        json.dump(meta_data, f)


def learn_pca(metadata, name, features_dir):
    """Computes regional mac features and learns PCA on them to be able 
    to whiten them later.

    This method assumes that there is enough memory available to hold all 
    r_macs and do the PCA computation. If this assumption does not hold, 
    we can switch to a memmapped numpy array and IncrementalPCA.
    """
    r_macs = []
    for idx in range(len(metadata)):
        data = metadata[str(idx)]
        features_file = join(features_dir, 
                                     'features/', 
                                     os.path.basename(data['image']))
        features = np.load('{}.npy'.format(features_file))
        r_macs += compute_r_macs(features)

    r_macs = np.vstack(r_macs)

    assert len(r_macs.shape) == 2
    assert r_macs.shape[0] >= r_macs.shape[1], \
        'Need at least {} rmacs to compute PCA, ' \
        'i.e. increase the number of input images'.format(r_macs.shape[1])

    print('Extracted {} rmacs on {} images'.format(r_macs.shape[0], 
                                                   len(metadata)))
    
    pca = PCA(n_components=r_macs.shape[1])
    pca.fit(r_macs)

    pca_path = join(features_dir, '{}.pca'.format(name))
    joblib.dump(pca, pca_path)
    print('Computed PCA and saved it to {}'.format(pca_path))


def compute_global_representation(metadata, name, features_dir, pca=None):
    """Uses previously extracted features to compute an image representation 
    which is suitable for image retrieval.
    """
    repr_store = None
    for idx, data in metadata.items():
        features_file = join(features_dir, 'features/', 
                             os.path.basename(data['image']))
        features = np.load('{}.npy'.format(features_file))

        representation = compute_representation(features, pca)
        if repr_store is None:
            repr_store = np.empty((len(metadata), representation.shape[-1]))
        repr_store[int(idx)] = np.squeeze(representation, axis=0)

    repr_file_path = join(features_dir, '{}.repr.npy'.format(name))
    np.save(repr_file_path, repr_store)
    print('Computed {} image representations and '
          'saved them to {}'.format(len(metadata), repr_file_path))


def main(args):
    args = parser.parse_args(args)

    if args.root_dir:
        args.root_dir = os.path.abspath(args.root_dir)

    if not os.path.exists(args.features_dir):
        os.mkdir(args.features_dir)

    if args.command == 'features':
        extract_conv_features(args.name, args.model, args.features_dir, 
                              args.image_dir, args.root_dir)
        return


    meta_file_path = join(args.features_dir, '{}.meta'.format(args.name))
    if not exists(meta_file_path):
        print('Must extract features of images first')
        return

    with open(meta_file_path, 'r') as f:
        metadata = json.load(f)

    if args.command == 'pca':
        learn_pca(metadata, args.name, args.features_dir)
    elif args.command == 'repr':
        pca_path = join(args.features_dir, '{}.pca'.format(args.name))
        if exists(pca_path):
            pca = joblib.load(pca_path)
        else:
            pca = None
        compute_global_representation(metadata, args.name, 
                                      args.features_dir, pca)
    

if __name__ == '__main__':
    main(sys.argv[1:])
