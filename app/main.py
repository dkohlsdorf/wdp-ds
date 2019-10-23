import os
import json
import db

from flask import Flask
from flask import jsonify
app = Flask(__name__)

@app.route('/')
def landing_page():
    return "Wild Dolphin Project Data Science"

@app.route("/wdp/encodings")
def encodings():
    response = app.response_class(
        response=json.dumps(db.encodings()),
        mimetype='application/json'
    )
    return response

@app.route("/wdp/pvl/encoding/<enc_id>")
def encoding(enc_id):
    response = app.response_class(
        response=json.dumps(db.pvl(enc_id)),
        mimetype='application/json'
    )
    return response

if __name__ == "__main__":
    app.run()
