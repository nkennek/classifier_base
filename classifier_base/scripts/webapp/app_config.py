#!/usr/bin/env python
# -- coding:utf-8 --

import os


class AppConfig(object):
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    ALLOWED_EXTENSIONS = set(['.png', '.jpg'])
    SECRET_KEY = os.urandom(24)

    def __init__(self):
        if not os.path.exists(self.UPLOAD_FOLDER):
            os.mkdir(self.UPLOAD_FOLDER)

config = AppConfig()
