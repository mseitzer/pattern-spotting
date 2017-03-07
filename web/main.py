#!/usr/bin/env python3
import sys
import os
import io
import time
import json
import argparse
from math import isnan

import requests
from PIL import Image
from flask import Flask, request, render_template, jsonify

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath(os.path.split(os.path.realpath(__file__))[0]
                                + '/..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.util import convert_image, crop_image
from src.search import SearchModel, search

MAX_FILE_SIZE = 16*1024*1024  # Maximum upload size 16MB
URL_TIMEOUT = 5  # Maximum time in seconds to wait for connection opening
DEFAULT_NUM_RESULTS = 5

app = Flask('Historical object retrieval')
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

parser = argparse.ArgumentParser(description=
                                 'Historical object retrieval web backend')
parser.add_argument('--config', default='config.txt',
                    help='Configuration file containing database paths')
parser.add_argument('-d', '--debug', action='store_true', 
                    help='Activate debug mode')
parser.add_argument('--test', action='store_true', 
                    help='Activate test mode')
parser.add_argument('--host', default='localhost',
                    help='Host address to listen on')

class InvalidUsage(Exception):
    def __init__(self, message, status_code=400, payload=None):
        Exception.__init__(self)
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        res = dict(self.payload or ())
        res['message'] = self.message
        return res


def download_file(url):
    """Downloads a file from a URL into a BytesIO object"""
    try:
        r = requests.get(url, stream=True, timeout=URL_TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    if int(r.headers.get('Content-Length')) > MAX_FILE_SIZE:
        return None

    file = io.BytesIO()
    size = 0
    start = time.time()
    for chunk in r.iter_content(1024):
        if time.time() - start > 5 * URL_TIMEOUT:
            return None

        size += len(chunk)
        if size > MAX_FILE_SIZE:
            return None

        file.write(chunk)

    return file


def parse_search_parameters(form):
    if 'x1' not in form or 'y1' not in form \
        or 'x2' not in form or 'y2' not in form:
        raise InvalidUsage('Missing bounding box parameter', 400)

    try:
        bounding_box = (int(form['x1']), int(form['y1']),
                        int(form['x2']), int(form['y2']))
    except ValueError:
        raise InvalidUsage('Invalid bounding box parameter', 400)

    try:
        num_res = int(form['num_results'])
    except ValueError:
        num_res = DEFAULT_NUM_RESULTS
    
    if num_res <= 0 or num_res > 100:
        num_res = DEFAULT_NUM_RESULTS

    return bounding_box, num_res


def search_image(image, bounding_box, top_n):
    try:
        image = Image.open(image).convert('RGB')
    except (IOError, OSError):
        raise InvalidUsage('Error decoding image', 415)

    try:
        crop = crop_image(image, bounding_box)
    except ValueError as e:
        raise InvalidUsage('Bad bounding box', 400)

    try:
        indices, scores, bboxes = search(search_model, convert_image(crop),
                                         top_n=top_n)
    except ValueError as e:
        print('Error while searching for roi: {}'.format(e))
        if app.debug:
            import traceback
            traceback.print_tb(e.__traceback__)
        raise InvalidUsage('Internal error while searching', 400)

    # Build response
    results = []
    for index, score, bbox in zip(indices, scores, bboxes):
        image_path = search_model.get_metadata(index)['image']
        image_info = search_model.query_database(image_path)
        if image_info is None:
            print('Warning: result image {} not found in db'.format(image_path))
            continue
        image_dict = {
            'name': os.path.basename(image_path),
            'score': round(score, 4) if not isnan(score) else 'NaN',
            'url': 'static/data/' + image_path,
            'ext_url': image_info.get('external_url', ''),
            'bbox': {'x1': bbox[0], 'y1': bbox[1],
                     'x2': bbox[2], 'y2': bbox[3]}
        }
        results.append(image_dict)
    return results


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/")
def index():
    return render_template('search.html')


@app.route("/search_file", methods=['POST'])
def search_file():
    bounding_box, num_res = parse_search_parameters(request.form)

    if 'file' in request.files:  # Image upload
        img_file = request.files['file']
    else:
        raise InvalidUsage('No image source', 400)

    results = search_image(img_file, bounding_box, num_res)
    return jsonify(**{'results': results})


@app.route("/search_url", methods=['POST'])
def search_url():
    bounding_box, num_res = parse_search_parameters(request.form)

    if 'url' in request.form:  # External image
        url = request.form['url']
        img_file = download_file(url)
        if not img_file:
            raise InvalidUsage('Error downloading external image', 400)
    else:
        raise InvalidUsage('No image source', 400)

    results = search_image(img_file, bounding_box, num_res)
    return jsonify(**{'results': results})


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    if args.debug:
        app.debug = True

    if args.test:
        # In test mode, overwrite the default search implementation
        global search
        from mock_search import search

    with open(args.config, 'r') as f:
        config = json.load(f)

    search_model = SearchModel(config['model'],
                               config['features'],
                               config['database'])
    
    print('Server running with model "{}", features "{}" ' 
          'and image database "{}".'.format(config['model'], 
                                            config['features'], 
                                            config['database']))
    app.run(host=args.host)
