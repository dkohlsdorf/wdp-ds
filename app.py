import re
import datetime
import pickle
import flask
import flask_login

from decoder_worker import DiscoveryService
from flask import Flask, render_template

VERSION  = 'Mar2022' 
SEQ_PATH = f'../web_service/{VERSION}/sequences/'
IMG_PATH = f'../web_service/{VERSION}/images/'
PKL_PATH = f'../web_service/{VERSION}/service.pkl'

USERS    = {'dolphin-visitor': {'password' : 'stenella_frontalis'}}
SECRET   = 'wdp-ds-dolphin' 


try:
    DISCOVERY = pickle.load(open(PKL_PATH, "rb"))
except (OSError, IOError) as e:
    DISCOVERY = DiscoveryService(SEQ_PATH)
    pickle.dump(DISCOVERY, open(PKL_PATH, "wb"))
    
print(DISCOVERY.substrings.keys())

app = Flask(__name__,
            static_url_path = '', 
            static_folder   = IMG_PATH,
            template_folder = 'templates')
app.secret_key = SECRET


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


@app.route('/neighborhood/<key>')
@flask_login.login_required
def neighborhood(key):
    key       = int(key)
    s         = DISCOVERY.get(key)
    sequences = [process_sequence(s[0])] + [process_sequence(x) for x in s[1][1:]]
    return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = s[2])


@app.route('/find', methods=['POST'])
@flask_login.login_required
def find():
    string = flask.request.form['query']
    print(string)
    sequences, keys = DISCOVERY.find(string)
    sequences = [process_sequence(x) for x in sequences]
    return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = keys)
                       
                       
                       
