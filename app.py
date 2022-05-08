import re
import datetime
import pickle
import flask
import flask_login

import sqlite3
import os.path

from decoder_worker import DiscoveryService
from flask import Flask, render_template, flash, redirect, request


VERSION     = 'Test' 
SEQ_PATH    = f'../web_service/{VERSION}/sequences/'
IMG_PATH    = f'../web_service/{VERSION}/images/'
PKL_PATH    = f'../web_service/{VERSION}/service.pkl'
UPLOAD_PATH = f'../web_service/{VERSION}/wav'
MODEL_PATH  = '../web_service/ml_models_mai_smlr/'


LIMIT      = 50

USERS    = {'dolphin-visitor': {'password' : 'stenella_frontalis'}}
SECRET   = 'wdp-ds-dolphin' 

try:
    DISCOVERY = pickle.load(open(PKL_PATH, "rb"))
except (OSError, IOError) as e:
    DISCOVERY = DiscoveryService(SEQ_PATH, IMG_PATH, LIMIT)
    pickle.dump(DISCOVERY, open(PKL_PATH, "wb"))

DISCOVERY.init_model(MODEL_PATH)

class QueryHistory:
    
    def __init__(self, file='query.db'):
        if not os.path.exists(file):
            conn = sqlite3.connect('query.db')       
            cur  = conn.cursor()             
            cur.execute("""
                CREATE TABLE query_history (
                     query_string text,
                     query_file text,
                     date text
                )
            """)
            conn.commit()
            conn.close()
        
    def insert(self, query, file=None):
        conn = sqlite3.connect('query.db')       
        cur  = conn.cursor()             

        date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        cur.execute("INSERT INTO query_history VALUES (?, ?, ?)", (query, file, date))
        conn.commit()
        conn.close()

    def get(self, n=None):
        conn = sqlite3.connect('query.db')               
        cur  = conn.cursor()         
        filter_n = f"LIMIT {n}" if n is not None else ""   
        cur.execute(f"""
            SELECT * 
            FROM query_history
            ORDER BY date DESC
            {filter_n}
        """)
        result = cur.fetchall()
        conn.commit()
        conn.close()        
        return result



app = Flask(__name__,
            static_url_path = '', 
            static_folder   = IMG_PATH,
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
    id       =  re.sub("[^a-zA-Z0-9 ]+", "", id)
    time     = f"{str(datetime.timedelta(seconds=s['start'] / 44100)) } - {str(datetime.timedelta(seconds=s['stop']  / 44100)) }" 
    start    = s['start']
    stop     = s['stop']        
    img      = f"{id}_{start}_{stop}.png"
    sequence = " ".join([c['cls'] for c in s['sequence']])
    
    return {
        "id"      : id,
        "start"   : start,
        "stop"    : stop,
        "img"     : img,
        "sequence": sequence,
        "time"    : time
    }


@app.route('/discovery')
@flask_login.login_required
def discovery():
    s         = DISCOVERY.sample()
    sequences = [process_sequence(s[0])] + [process_sequence(x) for x in s[1][1:]]
    return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = s[2])


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
        img, decoding, nn, keys = DISCOVERY.query_by_file(path)
        decoding = " ".join(decoding)
        
        history = QueryHistory()
        history.insert(decoding, path.split('/')[-1])

        sequences = [process_sequence(x) for x in nn]        
        return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = keys, query=(img, decoding))
    
    
@app.route('/neighborhood/<key>')
@flask_login.login_required
def neighborhood(key):
    key       = int(key)
    s         = DISCOVERY.get(key)
    sequences = [process_sequence(s[0])] + [process_sequence(x) for x in s[1][1:]]
    return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = s[2])


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
    return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = keys)

