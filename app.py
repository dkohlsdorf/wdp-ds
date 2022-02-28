import re
import datetime

from decoder_worker import DiscoveryService
from flask import Flask, render_template


SEQ_PATH = '../web_service/sequences/'
IMG_PATH = '../web_service/images/'

DISCOVERY = DiscoveryService(SEQ_PATH)

app = Flask(__name__,
            static_url_path = '', 
            static_folder   = IMG_PATH,
            template_folder = 'templates')


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
def discovery():
    s         = DISCOVERY.sample()
    sequences = [process_sequence(s[0])] + [process_sequence(x) for x in s[1][1:]]
    return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = s[2])


@app.route('/neighborhood/<key>')
def neighborhood(key):
    key       = int(key)
    s         = DISCOVERY.get(key)
    sequences = [process_sequence(s[0])] + [process_sequence(x) for x in s[1][1:]]
    return render_template('discovery.html', sequences=sequences, n=len(sequences), keys = s[2])
