import sys
import os
from flask import Flask, request, render_template

# Path hack to be able to import from sibling directory
sys.path.append(os.path.abspath('../src'))
#from models import load


app = Flask(__name__)
app.debug = True

@app.route("/")
def index():
    return render_template('search.html')

@app.route("/search", methods=['POST'])
def test():
    print(request.form['x1'])
    print(request.form['y1'])
    if request.method == 'GET':
        return "444"
    else:
        return "88"

if __name__ == "__main__":
    app.run()
