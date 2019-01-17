# -*- coding: utf-8 -*-

import os
import sqlite3
import re
import base64
import io
from datetime import datetime

from sqlite_tensor import Database, Tensor
from flask import Flask, jsonify, request, abort
from flask.json import JSONEncoder
import numpy as np


class NPArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Tensor):
            return {
                'id': obj.id,
                'data': self.default(obj.data),
                'attr': self.default(obj.attr)
            }
        if isinstance(obj, np.lib.npyio.NpzFile):
            return {
                k: self.default(obj[k]) for k in obj.keys()
            }
        if isinstance(obj, np.ndarray):
            return {
                'shape': obj.shape,
                'array': obj.tolist()
            }
        return obj


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
            q = request.get_json()
            t = Tensor(
                data=np.load(
                    io.BytesIO(
                        base64.b64decode(
                            re.sub('.*,', '', q['data'])
                        )
                    )
                ),
                attr=dict({
                    k: v for k, v in q.items() if k not in ('data', 'id')
                }, timestamp=datetime.utcnow().timestamp()),
                id=(q['id'] if 'id' in q else None)
            )
            db.save(t)
            return jsonify(t)

    @app.route('/api/nparray/<id_>', methods=['GET'])
    def api_nparray_id(id_):
        if request.method == 'GET':
            if id_ in db:
                t = db[id_]
                return jsonify(t)
            else:
                abort(404)

    return app
