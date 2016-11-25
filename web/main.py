import sys
import os
import io
import time
import requests
from PIL import Image
from flask import Flask, request, render_template, jsonify

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath('../src'))
#from models import load

MAX_FILE_SIZE = 16*1024*1024  # Maximum upload size 16MB
URL_TIMEOUT = 5  # Maximum time in seconds to wait for connection opening

app = Flask(__name__)
app.debug = True
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

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
    if 'file' in request.files:
        img_file = request.files['file']
    elif 'url' in request.form:
        url = request.form['url']
        img_file = download_file(url)
        if not img_file:
            raise InvalidUsage('Error downloading external image', 400)

    if not img_file:
        raise InvalidUsage('No image source', 400)

    try:
        img = Image.open(img_file)
    except (IOError, OSError):
        raise InvalidUsage('Error decoding image', 415)

    # TODO: implement search on bounding box in src/search/search.py
    # Maybe wrap database obj etc. in class
    return "test"

if __name__ == "__main__":
    # TODO: load databases on startup
    app.run()
