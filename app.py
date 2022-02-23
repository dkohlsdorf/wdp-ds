from decoder_worker import DiscoveryService
from flask import Flask, render_template


SEQ_PATH = '../web_service/sequences/'
IMG_PATH = '../web_service/images/'

DISCOVERY = DiscoveryService(SEQ_PATH)

app = Flask(__name__,
            static_url_path = '', 
            static_folder   = IMG_PATH,
            template_folder = 'templates')


def image_path(path, start, stop):
    image = path.split('/')[-1].replace('.wav', '')    
    return f"{image}_{start}_{stop}.png"


@app.route('/discovery')
def discovery():
    s = DISCOVERY.sample()
    return render_template('discovery.html', sequences=s)
