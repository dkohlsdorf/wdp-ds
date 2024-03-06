import re
import datetime
import pickle
import flask
import flask_login

import os.path

from redis import Redis
from decoder_worker import DiscoveryService, DecodingWorker
from dbs import *
from flask import Flask, render_template, flash, redirect, request
from lib_dolphin.parameters import *


VERSION = 'extern_clean' 
SEQ_PATH = f'static/web_service/{VERSION}/sequences/'
IMG_PATH = f'static/web_service/{VERSION}/images/'
PKL_PATH = f'static/web_service/{VERSION}/service.pkl'
UPLOAD_PATH = f'static/web_service/{VERSION}/wav/'
ALIGNMENT_UPLOAD_PATH = f'static/web_service/{VERSION}/alignment/'

DISABLE_SERVICE = True

LIMIT = None

USERS = {'dolphin-visitor': {'password' : 'stenella_frontalis'}}
SECRET = 'wdp-ds-dolphin' 

try:
    print(f"... Loading {PKL_PATH}")
    DISCOVERY = pickle.load(open(PKL_PATH, "rb"))
    print("... done")
except (OSError, IOError) as e:
    DISCOVERY = DiscoveryService(SEQ_PATH, IMG_PATH, LIMIT)
    pickle.dump(DISCOVERY, open(PKL_PATH, "wb"))

if not DISABLE_SERVICE:
    print("... init service")
    DISCOVERY.init_model(MODEL_PATH)
    print("... done")
      

app = Flask(__name__,
            static_url_path = '', 
            static_folder   = 'static',
            template_folder = 'templates')
app.secret_key = SECRET
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH 


login_manager = flask_login.LoginManager()
login_manager.init_app(app)


class User(flask_login.UserMixin):
    pass
    
    
@login_manager.user_loader
def user_loader(email):
    if email not in USERS:
        return
    user = User()
    user.id = email
    return user


@login_manager.request_loader
def request_loader(request):
    email = request.form.get('email')
    if email not in USERS:
        return
    user = User()
    user.id = email
    return user


@app.route('/', methods=['GET', 'POST'])
def login():
    if flask.request.method == 'GET':
        return '''
               <h1> [:WDP:] Dolphin Discovery Tool </h1>
               <form action='/' method='POST'>
                <input type='text' name='email' id='email' placeholder='email'/>
                <input type='password' name='password' id='password' placeholder='password'/>
                <input type='submit' name='submit'/>
               </form>
               '''    
    email = flask.request.form['email']
    if flask.request.form['password'] == USERS[email]['password']:
        user = User()
        user.id = email
        flask_login.login_user(user)
        return flask.redirect(flask.url_for('discovery'))
    return 'Bad login'


def process_sequence(s):
    id       = s['path'].split("/")[-1].replace('.wav', "")
    id       = re.sub("[^a-zA-Z0-9_ ]+", "", id)
    time     = f"{str(datetime.timedelta(seconds=s['start'] / 44100)) } - {str(datetime.timedelta(seconds=s['stop']  / 44100)) }" 
    start    = s['start']
    stop     = s['stop']        

    img      = f"{id}_{start}_{stop}.png"
    audio    = f"{id}_{start}_{stop}.wav"
    raven    = f"{id}_{start}_{stop}.txt"
    
    sequence = " ".join([c['cls'] for c in s['sequence']])
    
    return {
        "id"      : id,
        "start"   : start,
        "stop"    : stop,
        "img"     : img,
        "audio"   : audio,
        "raven"   : raven, 
        "sequence": sequence,
        "time"    : time
    }


@app.route('/discovery')
@flask_login.login_required
def discovery():
    s         = DISCOVERY.sample()
    sequences = [process_sequence(s[0])] + [process_sequence(x) for x in s[1][1:]]
    return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = s[2], version=VERSION)


@app.route('/alignment_seq/<file_id>')
@flask_login.login_required
def alignment_seq(file_id):
    path = f"{ALIGNMENT_UPLOAD_PATH}/{file_id}.json"
    with open(path) as f:
        data = f.read()
    return data


@app.route('/alignment_project/<project_id>', methods=['POST', 'GET'])
@flask_login.login_required
def alignment_project(project_id):
    db = AlignmentDB()
    project_id = int(project_id)
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            flash('No File Uploaded')
            return redirect(f'/alignment_project/{project_id}')            
        file = request.files['file']
        if not file.filename.endswith('.wav'):
            flash('Only wav files are allowed')
            return redirect(f'/alignment_project/{project_id}')

        path = f"{ALIGNMENT_UPLOAD_PATH}/{file.filename}"
        file.save(path)
        
        db.insert_file(project_id, file.filename.split("/")[-1].replace(".wav", ""))        
        print("Done Upload")
        print(f" .... redis {DecodingWorker.JSON_KEY} {path}")
        r = Redis()        
        r.lpush(DecodingWorker.JSON_KEY, path)
        return flask.redirect(flask.url_for('alignment_project', project_id=project_id))
    else:
        files = db.get_files(project_id)
        n = len(files)
        return render_template('alignment_project.html', project_id=project_id, files=files, n=n, version=VERSION)


@app.route('/query_relaxed', methods=['POST'])
@flask_login.login_required
def upload_relaxed():
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            flash('No File Uploaded')
            return redirect('/discovery')            
        file = request.files['file']
        if not file.filename.endswith('.wav'):
            flash('Only wav files are allowed')
            return redirect('/discovery')
        path = f"{UPLOAD_PATH}/{file.filename}"
        file.save(path)
        print("Done Upload")
        img, decoding, nn, keys = DISCOVERY.query_by_file(path, True)
        decoding = " ".join(decoding)
        
        history = QueryHistory()
        history.insert(decoding, path.split('/')[-1])

        sequences = [process_sequence(x) for x in nn]        
        return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = keys, query=(img, decoding), version=VERSION)
    

@app.route('/query', methods=['POST'])
@flask_login.login_required
def upload():
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            flash('No File Uploaded')
            return redirect('/discovery')            
        file = request.files['file']
        if not file.filename.endswith('.wav'):
            flash('Only wav files are allowed')
            return redirect('/discovery')
        path = f"{UPLOAD_PATH}/{file.filename}"
        file.save(path)
        print("Done Upload")
        img, decoding, nn, keys = DISCOVERY.query_by_file(path, False)
        decoding = " ".join(decoding)
        
        history = QueryHistory()
        history.insert(decoding, path.split('/')[-1])

        sequences = [process_sequence(x) for x in nn]        
        return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = keys, query=(img, decoding), version=VERSION)
    
    
@app.route('/neighborhood/<key>')
@flask_login.login_required
def neighborhood(key):
    key       = int(key)
    s         = DISCOVERY.get(key)
    sequences = [process_sequence(s[0])] + [process_sequence(x) for x in s[1][1:]]
    return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = s[2], version=VERSION)


@app.route('/history')
@flask_login.login_required
def history():
    history = QueryHistory()
    h = history.get()
    n = len(h)
    return render_template('history.html', rows = h, n = n)


@app.route('/find', methods=['POST', 'GET'])
@flask_login.login_required
def find():
    if request.method == 'POST':    
        string = flask.request.form['query']
        history = QueryHistory()
        history.insert(string)
    else:
        string = request.args.get('strg', None)
    sequences, keys = DISCOVERY.find(string)
    sequences = [process_sequence(x) for x in sequences]
    return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = keys, strg = string, version=-VERSION)

