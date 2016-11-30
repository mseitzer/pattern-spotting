#!/usr/bin/env python3
import sys
import os
import io
import time
import json
import argparse

import requests
from PIL import Image
from flask import Flask, request, render_template, jsonify

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath(os.path.split(os.path.realpath(__file__))[0]
                                + '/..'))
from src.search import SearchModel, search_roi

MAX_FILE_SIZE = 16*1024*1024  # Maximum upload size 16MB
URL_TIMEOUT = 5  # Maximum time in seconds to wait for connection opening

app = Flask('Historical object retrieval')
app.debug = True
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

parser = argparse.ArgumentParser(description=
                                 'Historical object retrieval web backend')
parser.add_argument('--config', default='config.txt',
                    help='Configuration file containing database paths')

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


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/")
def index():
    return render_template('search.html')


@app.route("/search", methods=['POST'])
def search():
    if 'x1' not in request.form or 'y1' not in request.form \
        or 'x2' not in request.form or 'y2' not in request.form:
        raise InvalidUsage('Missing bounding box parameter', 400)

    try:
        bounding_box = (int(request.form['x1']), int(request.form['y1']),
                        int(request.form['x2']), int(request.form['y2']))
    except ValueError:
        raise InvalidUsage('Invalid bounding box parameter', 400)

    img_file = None
    if 'file' in request.files:  # Image upload
        img_file = request.files['file']
    elif 'url' in request.form:  # External image
        url = request.form['url']
        img_file = download_file(url)
        if not img_file:
            raise InvalidUsage('Error downloading external image', 400)

    if not img_file:
        raise InvalidUsage('No image source', 400)

    try:
        image = Image.open(img_file)
    except (IOError, OSError):
        raise InvalidUsage('Error decoding image', 415)

    try:
        indices, scores = search_roi(search_model, image, bounding_box, 5)
    except ValueError as e:
        import traceback
        traceback.print_tb(e.__traceback__)
        print(e)
        raise InvalidUsage('Invalid bounding box parameter, internal', 400)

    # TODO: compute bounding boxes on result images

    # Build response
    res_list = []
    for index, score in zip(indices, scores):
        image_path = search_model.get_metadata(index)['image']
        image_info = search_model.query_database(image_path)
        if image_info is None:
            print('Warning: result image {} not found in db'.format(image_path))
            continue

        image_dict = {
            'image': image_path,
            'score': score
        }

        if 'url' in image_info:
            image_dict['url'] = image_info['url']

        res_list.append(image_dict)

    return jsonify(**{'results': res_list})

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    with open(args.config, 'r') as f:
        config = json.load(f)
        print(config)

    search_model = SearchModel(config['model'],
                               config['features'],
                               config['database'])
    app.run()
