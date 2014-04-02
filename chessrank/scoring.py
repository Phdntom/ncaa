from __future__ import division

import numpy as np

def game_score(point_diff, weight=15):

    return 1 / ( 1 + 10 ** (-point_diff / weight) )

def point_diff(game_score, weight=15):

    return -weight * np.log10( 1 / game_score - 1)

def log_loss(y, y_pred):

    ln = np.log
    
    return  -np.mean(y * ln(y_pred) + (1 - y) * ln(1 - y_pred))
