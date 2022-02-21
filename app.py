from decoder_worker import DiscoveryService
from flask import Flask

SEQ_PATH = '../web_service/sequences/'
IMG_PATH = '../web_service/images/'

DISCOVERY = DiscoveryService(SEQ_PATH)

app = Flask(__name__)


def image_path(path, start, stop):
    image = path.split('/')[-1].replace('.wav', '')    
    return f"{IMG_PATH}/{image}_{start}_{stop}.png"

@app.route('/sequences')
def sequences():    
    s = [(s['path'], s['start'], s['stop'], image_path(s['path'], s['start'], s['stop'])) for s in DISCOVERY.sequences]
    return str(s)

@app.route('/discovery')
def discovery():
    s = DISCOVERY.sample()
    return str(s)
