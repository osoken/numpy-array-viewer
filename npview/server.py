# -*- coding: utf-8 -*-

import os
import sqlite3
import re
import base64
import io
import json

from sqlite_tensor import Database, Tensor
from flask import Flask, jsonify, request, abort
from flask.json import JSONEncoder
import numpy as np


class NPArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Tensor):
            return {
                'id': obj.id,
                'data': self.default(obj.data)
            }
        if hasattr(obj, 'keys'):
            return {
                k: self.default(obj[k]) for k in obj.keys()
            }
        if isinstance(obj, np.ndarray):
            return {
                'shape': obj.shape,
                'tensor': obj.tolist()
            }
        return json.JSONEncoder.default(self, obj)


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

    app.json_encoder = NPArrayEncoder

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

    @app.route('/api/nparray/<id_>', methods=['GET'])
    def api_nparray_id(id_):
        if request.method == 'GET':
            if id_ in db:
                t = db[id_]
                return jsonify(t)
            else:
                abort(404)

    return app
