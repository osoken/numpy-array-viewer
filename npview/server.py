# -*- coding: utf-8 -*-

import os
import sqlite3
import re
import base64
import io

from sqlite_tensor import Database, Tensor
from flask import Flask, jsonify, request
import numpy as np


def gen_app(config_object=None):

    app = Flask(__name__)
    app.config.from_object('npview.config')
    if os.getenv('NPVIEW_CONFIG') is not None:
        app.config.from_envvar('NPVIEW_CONFIG')
    if config_object is not None:
        app.config.update(**config_object)

    conn = sqlite3.connect(
        app.config['DATABASE_FILE_PATH'], check_same_thread=False
    )

    db = Database(conn)

    @app.route('/api/nparray', methods=['GET', 'POST'])
    def api_nparray():
        if request.method == 'GET':
            return jsonify({
                'result': list(db.keys())
            })
        elif request.method == 'POST':
            data = request.get_json()
            t = Tensor(
                np.load(
                    io.BytesIO(
                        base64.b64decode(
                            re.sub('.*,', '', data['data'])
                        )
                    )
                )
            )
            db.save(t)
            return jsonify({
                'id': t.id
            })

    return app
