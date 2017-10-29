#!/usr/bin/env python
# -- coding:utf-8 --

import argparse
import os
import sys

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from skimage import io

from classifier_base.scripts.webapp.app_config import config as constants
from classifier_base.scripts.models.predict import Predictor
from classifier_base.scripts.models.model_config import ModelConfig

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = constants.UPLOAD_FOLDER
app.config['SECRET_KEY'] = constants.SECRET_KEY
# setup for classifier
clf_config = ModelConfig(base_dir=os.path.dirname(__file__))
clf = Predictor(clf_config, model_path='newest')


def allowed_file(filename, exts):
    _, ext = os.path.splitext(filename)
    return ext in exts


def build_new_filename(orig_filename):
    file_id = len(os.listdir(app.config['UPLOAD_FOLDER'])) + 1
    _, ext = os.path.splitext(orig_filename)
    filename = 'uploads_{}'.format(file_id) + ext
    return filename


def filename_for_client(filename):
    return os.path.join('/uploads', os.path.basename(filename))


@app.route('/')
def index():
    return render_template('index.html'), 200


@app.route('/send', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return redirect(url_for('index'))
    img_file = request.files['img_file']
    if not img_file or not allowed_file(img_file.filename, constants.ALLOWED_EXTENSIONS):
        return ''' <p>許可されていない拡張子です</p> ''', 400

    filename = build_new_filename(img_file.filename)
    img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_file.save(img_url)
    pred = clf(io.imread(img_url, 0))
    judge = clf_config.category[np.argmax(pred)]
    confidence = max(pred)
    return render_template('index.html', img_url=filename_for_client(img_url), judge=judge, confidence=100.*confidence), 200


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename), 200


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch Application Server")
    parser.add_argument('--port', action='store', type=int, nargs='?', default=80)
    args = parser.parse_args()
    app.debug=True
    # https://qiita.com/tomboyboy/items/122dfdb41188176e45b5
    app.run(host='0.0.0.0', port=args.port)
