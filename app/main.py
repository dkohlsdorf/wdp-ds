import os
import json
import db
import tempfile
import datetime

from google.cloud import storage
import base64

from flask import Flask, jsonify, request, redirect
app = Flask(__name__)

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

@app.route("/wdp/asset/cluster_files/<algorithm>")
def cluster_files(algorithm):    
    def is_cluster(path):
        name = path.split("/")[-1]
        return name.startswith('cluster') and name.endswith('.wav')
    def to_dict(path):
        name = path.split("/")[-1]
        return {"name":name}            
    client = storage.Client.from_service_account_json('secret.json')
    bucket = client.get_bucket('wdp-data')
    blist  = bucket.list_blobs(prefix=algorithm)
    assets = [to_dict(f) for f in blist if is_cluster(f)] 
    response = app.response_class(
        response=json.dumps(assets),
        mimetype='application/json'
    )
    return response

@app.route("/wdp/asset/<algorithm>/<assetname>")
def get_asset(algorithm,assetname):
    path = "{}/{}".format(algorithm, assetname)
    client = storage.Client.from_service_account_json('secret.json')
    bucket = client.get_bucket('wdp-data')
    blob   = bucket.blob(path)
    url = blob.generate_signed_url(
        version='v4',
        expiration=datetime.timedelta(minutes=1),
        method='GET')
    return redirect(url, code=302)

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
    
@app.route("/wdp/ds/<algorithm>/<enc_id>")
def clusters(enc_id, algorithm):
    response = app.response_class(
        response=json.dumps(db.clusters(enc_id, algorithm)),
        mimetype='application/json'
    )
    return response

@app.route("/wdp/ds/algorithms")
def algorithms():
    response = app.response_class(
        response=json.dumps(db.algorithms()),
        mimetype='application/json'
    )
    return response
    
if __name__ == "__main__":
    app.run()
