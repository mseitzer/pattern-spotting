import os, sys
import argparse
import numpy as np

from features import compute_representation, load_image, database
from models import load

parser = argparse.ArgumentParser(description=
                                 'Query a database for similar images')
parser.add_argument('--database-dir', dest='database_dir',
                    help='Folder where databases are stored',
                    default='../database')
parser.add_argument('--database',
                    help='Name of the database to use')
parser.add_argument('--model-dir', dest='model_dir',
                    help='Folder where trained models are stored',
                    default='../models')
parser.add_argument('images', nargs='+', 
                    help='One or more images to query for')


def query(query_features, feature_store, top_n=0):
    """Returns a list of indices of the most similar features"""
    query_features = np.expand_dims(query_features, axis=0)

    # Cosine similarity measure
    similarity = feature_store.dot(query_features.T).flatten()

    if top_n == 0:
        k = -len(similarity)+1  # Every feature
    elif top_n > 0:
        k = -top_n+1  # Only best top_n
    else:
        k = -top_n  # Only worst top_n
    
    # TODO: return only top_n results
    indices = np.argpartition(similarity, k)
    indices = indices[np.argsort(similarity[indices])]  # Sort indices

    top_similarities = similarity[indices]
    return indices, top_similarities


def main(args):
    args = parser.parse_args(args)

    query_images = []
    for image_path in args.images:
        if os.path.exists(image_path):
            query_images.append(image_path)
        else:
            print('Image {} does not exist. Skipping.'.format(image_path))

    db_path = os.path.join(args.database_dir, args.database)
    db = database.load('{}.pkl'.format(db_path))
    feature_store = np.load('{}.npy'.format(db_path))

    model, preprocess_fn = load(db.model_name, args.model_dir)

    for image_path in query_images:
        image = load_image(image_path)
        image = preprocess_fn(image)

        features = compute_representation(model, image)

        top_n = 2
        top_results, top_similarities = query(features, feature_store, top_n)

        print('Top {} results for query image {}'.format(len(top_results), 
                                                         image_path))
        for result, similarity in zip(top_results, top_similarities):
            result_path = db.images[result]['path']
            print('{} - {}'.format(result_path, similarity))

if __name__ == '__main__':
    # Note: run from src/ with python3 -m search.search
    main(sys.argv[1:])
