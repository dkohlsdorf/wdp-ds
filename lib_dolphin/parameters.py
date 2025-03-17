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


# Detection Params
DETECTION_TH   = 0.75
SMOOTH_WIN     = 3
MIN_REGION_SZE = 4
