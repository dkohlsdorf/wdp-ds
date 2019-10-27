import os
import json
import db
import tempfile
import datetime

from google.cloud import storage
import base64

from flask import Flask, jsonify, request, redirect
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

@app.route("/wdp/wav/<enc_id>")
def read_file(enc_id):
    f    = db.filename(enc_id)[0]
    path = "audio_files/{}".format(f)
    client = storage.Client.from_service_account_json('secret.json')
    bucket = client.get_bucket('wdp-data')
    blob   = bucket.blob(path)
    url = blob.generate_signed_url(
        version='v4',
        expiration=datetime.timedelta(minutes=1),
        method='GET')
    return redirect(url, code=302)

@app.route("/wdp/ds/non_silent/<enc_id>")
def non_silent(enc_id):
    response = app.response_class(
        response=json.dumps(db.not_silent_regions(enc_id)),
        mimetype='application/json'
    )
    return response
    
if __name__ == "__main__":
    app.run()
