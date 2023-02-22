import numpy as np
from collections import defaultdict

# AUDIO PARAMS
FFT_STEP     = 128
FFT_WIN      = 512
FFT_HI       = 230
FFT_LO       = 100

D            = FFT_WIN // 2 - FFT_LO - (FFT_WIN // 2 - FFT_HI)
RAW_AUDIO    = 5120
T            = int((RAW_AUDIO - FFT_WIN) / FFT_STEP)


# ==========================
MODEL_PATH = '../web_service/ml_models_v2/'
NEURAL_NOISE_DAMPENING=5.0
NEURAL_LABEL_DAMPENING={
    'Uc' : 0.1,
    'Ea' : 0.0,
    'Eb' : 0.0,
    'Ec' : 0.0,
    'Ed' : 0.0,
    'Ee' : 0.0,
    'Ef' : 0.0,
    'Eg' : 0.0,    
    'Bc' : 0.1,
    'Be' : 0.1,    
    'Bf' : 0.1,
    'Bg' : 0.1,
}

NEURAL_REJECT= defaultdict(lambda: 0.1, {})
NEURAL_SMOOTH_WIN=128

"""
#SPOTTED DECODING PARAMS
MODEL_PATH = '../web_service/ml_models_nov9/'
NEURAL_NOISE_DAMPENING=0.5
NEURAL_LABEL_DAMPENING={
'Ea': 0.0,
'Eb': 0.0,
'Ec': 0.0,
'Ed': 0.0,
'Ee': 0.0,
'Ef': 0.0,
'Eg': 0.0,
'Eh': 0.0,
}

NEURAL_REJECT= defaultdict(lambda: 0.003, {
    'Bb': 0.33,
    'Bc': 0.33,
    'Bd': 0.75,
    'Be': 0.6,
})
NEURAL_SMOOTH_WIN=128 

# BOTTLENOSE DECODING PARAMS
MODEL_PATH = '../web_service/ml_models_v2/'
NEURAL_NOISE_DAMPENING=1.0
NEURAL_LABEL_DAMPENING={
'Ea': 0.0,
'Eb': 0.0,
'Ec': 0.0,
'Ed': 0.0,
'Ee': 0.0,
'Ef': 0.0,
'Eg': 0.0,
'Eh': 0.0,
'Dd': 0.0,
'Uc': 0.0,
'Ud': 0.0 
}

NEURAL_REJECT= defaultdict(lambda: 0.003, {
    'Be': 0.75,
    'Bc': 0.75

})
NEURAL_SMOOTH_WIN=128
"""
# ==========================


# MODEL PARAMS
CONV_PARAM   = [
    (8, 8,  32),
    (4, 16, 32),
    (2, 32, 32),
    (1, 64, 32),
    (8,  4, 32),
    (16, 4, 32),
    (32, 4, 32)
]

N_BANKS = len(CONV_PARAM)
N_FILTERS = int(np.sum([i for _, _, i in CONV_PARAM]))

WINDOW_PARAM = (T, D, 1)
LATENT       = 128
EPOCHS       = 10
BATCH        = 25


# SEQUENTIAL PARAMS
MIN_LEN = 44100 // 10
MAX_LEN = 44100 // 5

NEURAL_SIZE_TH = 0

SCALER = 1.0
BIAS   = 0.7
START  = 0.2
STOP   = 0.8

SPLIT_SEC    = 60
SPLIT_RATE   = 44100
SPLIT_SKIP   = 0.5
