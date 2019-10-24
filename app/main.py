import os
import json
import db

from wdp_iap import validate_assertion
from flask import , Flask, jsonify, redirect, request

app = Flask(__name__)

@app.route('/')
def landing_page():
    return "Wild Dolphin Project Data Science"

@app.route("/wdp/encodings")
def encodings():
    assertion = request.headers.get('X-Goog-IAP-JWT-Assertion')
    email, id = validate_assertion(assertion) 
    if email is not None and id is not None:
        response = app.response_class(
            response=json.dumps(db.encodings()),
            mimetype='application/json'
        )
        return response
    else:
        return redirect('wdp-ds.appspot.com', code='302')
        

@app.route("/wdp/pvl/encoding/<enc_id>")
def encoding(enc_id):
    assertion = request.headers.get('X-Goog-IAP-JWT-Assertion')
    email, id = validate_assertion(assertion) 
    if email is not None and id is not None:
        response = app.response_class(
            response=json.dumps(db.pvl(enc_id)),
            mimetype='application/json'
        )
        return response
    else:
        return redirect('wdp-ds.appspot.com', code='302')

if __name__ == "__main__":
    app.run()
