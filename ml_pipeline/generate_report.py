import yaml
import sys

from tensorflow.keras.models import *
from audio import *

def generate_from_model(config, file):
    '''
    Generate Model Description From File
    
    :param config: run config
    :param file: filename
    :returns: a model description string
    '''
    model = load_model("{}{}".format(config['output'], file))
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist[1:])
    return short_model_summary


def parameter_table(config):
    '''
    Generate a table with all the features
    
    :param config: run config
    :returns: markdown table
    '''
    return """
    |Parameter|Value|
    |:--- |:---|
    |DFT Win |{}|
    |DFT Skip|{}|
    |Spetrogram Win |{}|
    |Spetrogram Skip|{}|
    |Highpass|{}|
    |K-Means|{}|
    """.format(
        config['fft_win'],
        config['fft_step'],
        config['spec_win'],
        config['spec_step'],
        config['highpass'],
        config['k']
    ).replace("\t", "")
    
    
def image(config, file):
    '''
    Generate image link
    
    :param config: run config
    :param file: image file
    :returns: image link
    '''
    return "![{}]({})".format(file, file)

    
def from_template(filename, config):
    '''
    Write a filled template
    
    :param filename: template filename
    :param config: run configs
    '''
    template = open(filename).read()
    template = template.replace("<ENCODER>", generate_from_model(config, 'encoder.h5'))
    template = template.replace("<AUTO_ENCODER>", generate_from_model(config, 'auto_encoder.h5'))
    template = template.replace("<SIL>", generate_from_model(config, 'auto_encoder.h5'))
    template = template.replace("<VERSION>", config['version'])
    template = template.replace("<PARAMS>", parameter_table(config))
    template = template.replace("<CONFUSION>", image(config, "confusion.png"))
    template = template.replace("<CLUSTERING>", image(config, "embeddings_test.png"))
    with open('{}/Readme.md'.format(config['output']), "w") as fp:
        fp.write(template)
