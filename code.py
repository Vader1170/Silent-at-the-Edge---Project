import argparse
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from scipy.special import eval_legendre
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reliabilitychecker import (
    to_domain,
    trapezoid_weights,
    legendre_basis as legendre_design,
    fit_diagnostics,
    reliability_check as _rc_reliability_check,
    calibrated_thresholds,
    near_chebyshev_nodes as _rc_near_chebyshev_nodes,
    economic_benefit,
)
OUT = os.path.join(HERE, 'figs_dashboard')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

def gram_and_edge(t_grid, t_ref, m=10):
    d = fit_diagnostics(t_grid, t_ref=t_ref, m=m)
    kappa = d['kappa']
    var_edge = d['variance']
    leverage = d['leverage']
    return (kappa, var_edge, leverage)

def reliability_check(t_grid, m=10, t_ref=None, kappa_warn=1000.0, kappa_critical=1000000.0):
    return _rc_reliability_check(t_grid, t_ref=t_ref, m=m, kappa_warn=kappa_warn, kappa_critical=kappa_critical)

def register_station(key, name, t, wse=None, native_cadence_min=None, color='#555555', kind='REAL'):
    STATIONS[key] = lambda: dict(name=name, short=key, kind=kind, t=np.asarray(t, dtype=float), wse=None if wse is None else np.asarray(wse, dtype=float), native_cadence_min=native_cadence_min, color=color)

def chebyshev_like_nodes(n, alpha=-0.49, eps=0.001):
    return _rc_near_chebyshev_nodes(n, alpha=alpha, eps=eps)

def chebyshev_weight(x, alpha=-0.49, eps=0.05):
    w = (1 - x ** 2 + eps) ** alpha
    return w / w.mean()

def station_A():
    t_nominal = np.arange(0, 20 * 60 + 1, 15)
    missing_real = {240}
    t = np.array([x for x in t_nominal if x not in missing_real], dtype=float)
    return dict(name='Station A -- Potomac River, MD (USGS 01646500, free-flowing)', short='A', kind='REAL', t=t, wse=None, native_cadence_min=15.0, color='#B03A2E')

def station_B():
    t = np.arange(0, 23.75 * 60 + 1, 15.0)
    return dict(name='Station B -- Caloosahatchee River, FL (USGS 02292900, regulated)', short='B', kind='REAL', t=t, wse=None, native_cadence_min=15.0, color='#2E6FA7')
T_C_MIN = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140, 1200, 1260, 1320, 1380, 1440, 1500, 1560, 1620, 1680, 1740, 1800, 1860, 1920, 1980, 2040, 2100, 2160, 2220, 2280, 2340, 2400, 2460, 2520, 2580, 2640, 2700, 2760, 2820, 2880, 2940, 3000, 3060, 3120, 3180, 3240, 3300, 3360, 3420, 3480, 3540, 3600, 3660, 3720, 3780, 3840, 3900, 3960, 4020, 4080, 4140, 4200, 4260, 4320, 4380, 4440, 4500, 4560, 4620, 4680, 4740, 4800, 4860, 4920, 4980, 5040, 5100, 5160, 5220, 5280, 5340, 5400, 5460, 5520, 5580, 5640, 5700, 5760, 5820, 5880, 5940, 6000, 6060, 6120, 6180, 6240, 6300, 6360, 6420, 6480, 6540, 6600, 6660, 6720, 6780, 6840, 6900, 6960, 7020, 7080, 7140, 7200, 7260, 7320, 7380, 7440, 7500, 7560, 7620, 7680, 7740, 7800, 7860, 7920, 7980, 8040, 8100, 8160, 8220, 8280, 8340, 8400, 8460, 8520, 8580, 8640, 8700, 8760, 8820, 8880, 8940, 9000, 9060, 9120, 9180, 9240, 9300, 9360, 9420, 9480, 9540, 9600, 9660, 9720, 9780, 9840, 9900, 9960, 10020, 10080, 10140, 10200, 10260, 10320, 10380, 10440, 10500, 10560, 10620, 10680, 10740, 10800, 10860, 10920, 10980, 11040, 11100, 11160, 11220, 11280, 11340, 11400, 11460, 11520, 11580, 12060, 12360, 12660, 13500, 13800, 14100, 14940, 15240, 15540, 16380, 16680, 16980, 17340, 17820, 18120, 18420, 19260, 19560, 19860]
WSE_C = [219.82, 219.82, 219.81, 219.81, 219.81, 219.79, 219.79, 219.79, 219.79, 219.77, 219.77, 219.77, 219.77, 219.77, 219.77, 219.77, 219.77, 219.77, 219.77, 219.76, 219.76, 219.76, 219.76, 219.76, 219.75, 219.75, 219.75, 219.75, 219.75, 219.75, 219.75, 219.75, 219.75, 219.77, 219.77, 219.77, 219.77, 219.76, 219.76, 219.76, 219.75, 219.75, 219.75, 219.75, 219.75, 219.74, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.73, 219.73, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.72, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.75, 219.75, 219.75, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.74, 219.76, 219.76, 219.76, 219.76, 219.74, 219.74, 219.74, 219.74, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.72, 219.72, 219.72, 219.7, 219.7, 219.7, 219.7, 219.73, 219.73, 219.73, 219.73, 219.73, 219.73, 219.76, 219.76, 219.76, 219.76, 219.76, 219.76, 219.76, 219.81, 219.81, 219.8, 219.81, 219.8, 219.78, 219.76, 219.76, 219.75, 219.71, 219.71, 219.73, 219.0, 219.51, 219.43, 219.4, 219.73, 219.73, 219.73]

def station_C():
    t = np.array(T_C_MIN, dtype=float)
    wse = np.array(WSE_C, dtype=float)
    order = np.argsort(t)
    t, wse = (t[order], wse[order])
    return dict(name='Station C -- Ambarampalayam, Tamil Nadu (CWC 015-SRDCBE)', short='C', kind='REAL', t=t, wse=wse, native_cadence_min=60.0, color='#117864')

T_D_MIN = [0, 840, 1140, 1440, 2280, 2580, 2880, 3720, 4020, 4320, 5160, 5460, 5760, 6600, 6900, 7200, 8040, 8340, 8640, 9480, 9780, 10080, 10920, 11220, 11520, 12360, 12660, 12960, 13800, 14100, 14400, 15240, 15540, 15840, 16680, 16980, 17280, 18120, 18420, 18720, 19560, 19860, 20160, 21000, 21300, 21600, 22440, 22740, 23040, 23880, 24180, 24480, 25320, 25620, 25920, 26760, 27060, 27360, 28200, 28500, 28800, 29640, 29940, 30240, 31080, 31380, 31680, 32520, 32820, 33120, 33960, 34260, 34560, 35400, 35700, 36000, 36840, 37140, 37440, 38280, 38580, 38880, 39720, 40020, 40320, 41160, 41460, 41760, 42600, 42900, 43200, 44040, 44340, 44640, 45480, 45780, 46080, 46920, 47220, 47520, 48360, 48660, 48960, 49800, 50100, 50400, 51240, 51540, 51840, 52680, 52980, 53280, 54120, 54420, 54720, 55560, 55860, 56160, 57000, 57300, 57600, 58440, 58740, 59040, 59880, 60180, 60480, 61320, 61620, 61920, 62760, 63060, 63360, 64200, 64500, 64800, 65640, 65940, 66240, 67080, 67380, 67680, 68520, 68820, 69120, 69960, 70260, 70560, 71400, 71700, 72000, 72840, 73140, 73440, 74280, 74580, 74880, 75720, 76020, 76320, 77160, 77400, 77760, 78600, 78900, 79200, 80040, 80340, 80640, 81480, 81780, 82080, 82920, 83220, 83520, 84360, 84660, 84960, 85800, 86100, 86400, 87240, 87540, 87840, 88680, 88980, 89280, 90120, 90420, 90720, 91080, 91140, 91200, 91260, 91320, 91380, 91440, 91500, 91560, 91620]
WSE_D = [139.82, 140.15, 140.05, 140.05, 140.51, 139.9, 140.75, 140.6, 140.51, 140.45, 140.38, 140.31, 140.27, 140.22, 140.24, 140.27, 140.31, 140.33, 140.34, 140.37, 140.37, 140.39, 140.4, 140.42, 140.44, 140.47, 140.51, 140.58, 140.69, 140.64, 140.6, 140.55, 140.44, 140.32, 140.08, 140.14, 140.21, 140.34, 140.39, 140.44, 140.57, 140.41, 140.22, 140.0, 140.11, 140.2, 140.32, 140.19, 140.2, 139.8, 139.75, 139.71, 139.68, 139.83, 139.97, 140.2, 140.33, 140.41, 140.24, 140.4, 140.51, 140.3, 140.17, 140.31, 140.4, 140.21, 140.65, 140.76, 140.76, 140.75, 141.53, 141.02, 140.6, 140.47, 140.45, 140.36, 139.97, 140.0, 140.1, 140.31, 140.25, 140.36, 140.1, 140.38, 140.85, 139.87, 139.82, 139.7, 140.39, 140.24, 140.38, 140.25, 140.4, 140.46, 139.9, 140.03, 140.35, 139.97, 140.06, 140.28, 139.67, 139.8, 140.37, 139.71, 139.6, 140.0, 139.83, 139.66, 139.9, 139.71, 139.71, 140.21, 140.66, 141.02, 140.6, 139.68, 139.9, 140.71, 140.64, 140.64, 140.54, 139.72, 139.95, 140.29, 140.6, 140.2, 140.59, 139.65, 139.5, 139.45, 140.51, 140.7, 140.59, 140.42, 140.65, 140.34, 140.4, 140.67, 140.32, 139.77, 139.98, 140.15, 139.93, 140.17, 140.55, 140.05, 139.9, 140.2, 140.25, 140.4, 140.54, 140.37, 140.82, 140.44, 139.87, 140.15, 140.45, 140.32, 140.4, 140.4, 139.79, 140.18, 140.44, 139.75, 139.9, 140.33, 140.44, 140.18, 140.5, 140.11, 139.95, 140.52, 140.38, 140.77, 140.4, 139.76, 139.74, 139.92, 140.25, 140.32, 140.22, 140.28, 140.46, 140.26, 139.84, 139.8, 140.4, 141.4, 141.1, 140.9, 141.58, 141.58, 141.58, 141.58, 141.55, 141.53, 141.5, 141.47, 141.44, 141.4]

T_E_MIN = [0, 300, 600, 1440, 1740, 2040, 2880, 3180, 3480, 4320, 4620, 4920, 5760, 6060, 6360, 7200, 7500, 7800, 8640, 8940, 9240, 10080, 10380, 10680, 11520, 11820, 12120, 12960, 13260, 13560, 14400, 14700, 15000, 15840, 16140, 16440, 17280, 17580, 17880, 18720, 19020, 19320, 20160, 20460, 20760, 21600, 21900, 22200, 23040, 23340, 23640, 24480, 24780, 25080, 25920, 26220, 26520, 27360, 27660, 27960, 28800, 29100, 29400, 30240, 30540, 30840, 31680, 31980, 32280, 33120, 33420, 33720, 34560, 34860, 35160, 36000, 36300, 36600, 37440, 37740, 38040, 38880, 39180, 39480, 40320, 40620, 40920, 41760, 42060, 42360, 43200, 43500, 43800, 44640, 44940, 45240, 46080, 46380, 46680, 47520, 47820, 48120, 48960, 49260, 49560, 50400, 50700, 51000, 51840, 52140, 52440, 53280, 53580, 53880, 54720, 55020, 55320, 55740, 55800, 55860, 55920, 55980, 56040, 56100, 56160, 56220, 56280, 56340, 56400, 56460, 56520, 56580, 56640, 56700, 56760, 56820, 56880, 56940, 57000, 57060, 57120, 57180, 57240, 57300, 57360, 57420, 57480, 57540, 57600, 57660, 57720, 57780, 57840, 57900, 57960, 58020, 58080, 58140, 58200, 58260, 58320, 58380, 58440, 58500, 58560, 58620, 58680, 58740, 58800, 58860, 58920, 58980, 59040, 59100, 59160, 59220, 59280, 59340, 59400, 59460, 59520, 59580, 59640, 59700, 59760, 59820, 59880, 59940, 60000, 60060, 60120, 60180, 60240, 60300, 60360, 60420, 60480, 60540, 60600, 60660]
WSE_E = [208.1, 208.1, 208.1, 208.1, 208.1, 208.1, 208.1, 208.1, 208.1, 208.1, 208.1, 208.1, 208.1, 208.1, 208.26, 208.51, 208.51, 208.51, 208.51, 208.51, 208.51, 208.48, 208.48, 208.45, 208.41, 208.41, 208.3, 208.2, 208.2, 208.2, 208.2, 208.1, 208.1, 208.07, 208.07, 208.07, 208.07, 208.07, 208.07, 208.05, 208.05, 208.15, 208.4, 208.4, 208.4, 208.46, 208.4, 208.3, 208.26, 208.26, 208.24, 208.22, 208.21, 208.2, 208.19, 208.19, 208.19, 208.19, 208.19, 208.19, 208.2, 208.2, 208.55, 208.69, 208.69, 208.68, 208.66, 208.66, 208.4, 208.1, 208.1, 208.1, 208.1, 208.07, 208.07, 208.07, 208.07, 208.07, 208.06, 208.06, 208.06, 208.06, 208.06, 208.06, 208.06, 208.06, 208.06, 208.04, 208.04, 208.0, 207.97, 207.97, 207.97, 207.97, 207.95, 207.92, 207.9, 207.87, 207.87, 207.85, 207.85, 207.85, 207.83, 207.83, 208.13, 208.42, 208.42, 208.2, 208.04, 208.04, 208.04, 208.52, 208.52, 208.52, 208.52, 208.52, 208.52, 208.46, 208.4, 208.36, 208.33, 208.3, 208.26, 208.23, 208.2, 208.18, 208.18, 208.18, 208.18, 208.18, 208.18, 208.18, 208.18, 208.18, 208.18, 208.18, 208.18, 208.18, 208.18, 208.18, 208.18, 208.26, 208.3, 208.34, 208.38, 208.4, 208.42, 208.42, 208.42, 208.42, 208.42, 208.4, 208.4, 208.4, 208.38, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.36, 208.35, 208.35, 208.35, 208.35, 208.35, 208.35, 208.35, 208.35, 208.35, 208.35, 208.35, 208.35, 208.35, 208.35]

T_F_MIN = [0, 300, 600, 1440, 1740, 2040, 2880, 3180, 3480, 4320, 4620, 4920, 5760, 6060, 6360, 7200, 7500, 7800, 8640, 8940, 9240, 10080, 10380, 10680, 11520, 11820, 12120, 12960, 13260, 13560, 14400, 14700, 15000, 15840, 16140, 16440, 17280, 17580, 17880, 18720, 19020, 19320, 20160, 20460, 20760, 21600, 21900, 22200, 23040, 23340, 23640, 24480, 24780, 25080, 25920, 26220, 26520, 27360, 27660, 27960, 28800, 29100, 29400, 30240, 30540, 30840, 31680, 31980, 32280, 33120, 33420, 33720, 34560, 34860, 35160, 36000, 36300, 36600, 37440, 37740, 38040, 38880, 39180, 39480, 40320, 40620, 40920, 41760, 42060, 42360, 43200, 43500, 43800, 44640, 44940, 45240, 46080, 46380, 46680, 47520, 47820, 48120, 48960, 49260, 49560, 50400, 50700, 51000, 51840, 52140, 52440, 53280, 53580, 53880, 54720, 55020, 55320, 56040, 56160, 56460, 56760, 57600, 57900, 58200, 59040, 59340, 59640, 60480, 60780, 61080, 61920, 62220, 62520, 63360, 63660, 63960, 64800, 65100, 65400, 66240, 66540, 66840, 67680, 67980, 68280, 69120, 69420, 69720, 70560, 70860, 71160, 72000, 72060, 72300, 72540, 72600, 73440, 73740, 74040, 74880, 75180, 75480, 76320, 76620, 76920, 77760, 78060, 78180, 78360, 79200, 79500, 79800, 80640, 80940, 81240, 82080, 82380, 82680, 83520, 83820, 84120, 84960, 85260, 85560, 86400, 86700, 87000, 87840, 88140, 88440, 89280, 89580, 89880, 90720, 91020, 91320, 92160, 92460, 92760, 93180]
WSE_F = [97.36, 97.32, 97.33, 97.32, 97.34, 97.31, 97.28, 97.31, 97.3, 97.32, 97.3, 97.29, 97.33, 97.31, 97.3, 97.28, 97.26, 97.27, 97.29, 97.32, 97.29, 97.33, 97.33, 97.33, 97.31, 97.3, 97.33, 97.35, 97.38, 97.36, 97.44, 97.35, 97.34, 97.28, 97.29, 97.27, 97.3, 97.31, 97.31, 97.28, 97.27, 97.29, 97.32, 97.35, 97.33, 97.31, 97.33, 97.32, 97.29, 97.31, 97.36, 97.52, 97.58, 97.56, 97.6, 97.64, 97.65, 97.67, 97.7, 97.69, 97.71, 97.73, 97.75, 97.72, 97.74, 97.76, 97.75, 97.79, 97.74, 97.73, 97.65, 97.6, 97.5, 97.5, 97.48, 97.4, 97.39, 97.4, 97.39, 97.38, 97.36, 97.35, 97.36, 97.34, 97.39, 97.43, 97.42, 97.41, 97.42, 97.4, 97.39, 97.44, 97.3, 97.4, 97.45, 97.46, 97.45, 97.43, 97.4, 97.29, 97.31, 97.33, 97.3, 97.31, 97.33, 97.29, 97.31, 97.33, 97.31, 97.34, 97.35, 97.37, 97.41, 97.4, 97.38, 97.39, 97.42, 97.4, 97.4, 97.4, 97.46, 97.48, 97.5, 97.51, 97.49, 97.52, 97.51, 97.45, 97.47, 97.48, 97.46, 97.51, 97.5, 97.48, 97.49, 97.52, 97.53, 97.57, 97.55, 97.52, 97.53, 97.54, 97.51, 97.55, 97.53, 97.53, 97.56, 97.52, 97.48, 97.5, 97.51, 97.46, 97.46, 97.51, 97.49, 97.49, 97.45, 97.48, 97.5, 97.51, 97.55, 97.56, 97.54, 97.6, 97.61, 97.57, 97.59, 97.57, 97.59, 98.98, 98.95, 98.5, 98.3, 98.28, 98.2, 98.1, 98.08, 98.05, 98.02, 98.03, 98.01, 98.0, 98.02, 98.05, 98.01, 98.0, 97.98, 97.95, 97.94, 97.95, 97.9, 97.91, 97.89, 97.86, 97.86, 97.87, 97.83, 97.86, 97.87, 97.9]

T_G_MIN = [0, 840, 1140, 1440, 2280, 2580, 2880, 3720, 4020, 4320, 5160, 5460, 5760, 6600, 6900, 7200, 8040, 8340, 8640, 9480, 9780, 10080, 10920, 11220, 11520, 12360, 12660, 12960, 13800, 14160, 14400, 15240, 15540, 15840, 16680, 16980, 17280, 18120, 18420, 18720, 19560, 19860, 20160, 21000, 21300, 21600, 22440, 22740, 23040, 23880, 24180, 24480, 25320, 25620, 25920, 26760, 27060, 27360, 28200, 28500, 28800, 29640, 29940, 30240, 31080, 31380, 31680, 32520, 32820, 33120, 33960, 34260, 34560, 35400, 35700, 36000, 36840, 37140, 37440, 38280, 38580, 38880, 39720, 40020, 40320, 41160, 41460, 41760, 42600, 42900, 43200, 44040, 44340, 44640, 45480, 45780, 46080, 46920, 47220, 47520, 48360, 48660, 48960, 49800, 50100, 50400, 51240, 51300, 51360, 51420, 51480, 51540, 51600, 51660, 51720, 51780, 51840, 51900, 51960, 52020, 52080, 52140, 52200, 52260, 52320, 52380, 52440, 52500, 52560, 52620, 52680, 52740, 52800, 52860, 52920, 52980, 53040, 53100, 53160, 53220, 53280, 53340, 53400, 53460, 53520, 53580, 53640, 53700, 53760, 53820, 53880, 53940, 54000, 54060, 54120, 54420, 54720, 55560, 55860, 56160, 57000, 57300, 57600, 58440, 58740, 59040, 59880, 60180, 60480, 61320, 61620, 61920, 62760, 63060, 63360, 64200, 64500, 64800, 65640, 65940, 66240, 67080, 67380, 67680, 68520, 68820, 69120, 69960, 70260, 70560, 71400, 71700, 72420, 72480, 72540, 72600, 72660, 72720, 72780, 72840]
WSE_G = [203.3, 203.3, 203.3, 203.3, 203.3, 203.3, 203.3, 203.3, 203.3, 203.3, 203.3, 203.3, 203.3, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.29, 203.27, 203.27, 203.27, 203.29, 203.27, 203.27, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.25, 203.25, 203.25, 203.25, 203.25, 203.25, 203.25, 203.25, 203.25, 203.24, 203.24, 203.24, 203.24, 203.24, 203.24, 203.24, 203.24, 203.24, 203.24, 203.24, 203.24, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.23, 203.76, 203.7, 203.65, 203.6, 203.45, 203.45, 203.39, 203.39, 203.39, 203.39, 203.37, 203.37, 203.37, 203.37, 203.37, 203.37, 203.36, 203.36, 203.36, 203.35, 203.35, 203.35, 203.35, 203.35, 203.35, 203.33, 203.33, 203.33, 203.33, 203.33, 203.33, 203.31, 203.31, 203.31, 203.29, 203.29, 203.29, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.26, 203.24, 203.24]

def station_D():
    t = np.array(T_D_MIN, dtype=float)
    wse = np.array(WSE_D, dtype=float)
    order = np.argsort(t)
    t, wse = (t[order], wse[order])
    return dict(name='Station D -- Mandleshwar, Madhya Pradesh (CWC 017-NDBHP)', short='D', kind='REAL', t=t, wse=wse, native_cadence_min=300.0, color='#7D3C98')

def station_E():
    t = np.array(T_E_MIN, dtype=float)
    wse = np.array(WSE_E, dtype=float)
    order = np.argsort(t)
    t, wse = (t[order], wse[order])
    return dict(name='Station E -- Basantpur, Odisha (CWC 027-MDBURLA)', short='E', kind='REAL', t=t, wse=wse, native_cadence_min=300.0, color='#D4AC0D')

def station_F():
    t = np.array(T_F_MIN, dtype=float)
    wse = np.array(WSE_F, dtype=float)
    order = np.argsort(t)
    t, wse = (t[order], wse[order])
    return dict(name='Station F -- Basantpur_CWC (CWC 055-mgd2lkn)', short='F', kind='REAL', t=t, wse=wse, native_cadence_min=300.0, color='#229954')

def station_G():
    t = np.array(T_G_MIN, dtype=float)
    wse = np.array(WSE_G, dtype=float)
    order = np.argsort(t)
    t, wse = (t[order], wse[order])
    return dict(name='Station G -- Dhariawad, Rajasthan (CWC 003-MAHGAND)', short='G', kind='REAL', t=t, wse=wse, native_cadence_min=300.0, color='#CA6F1E')

STATIONS = {'A': station_A, 'B': station_B, 'C': station_C}
for _k, _f in [('D', station_D), ('E', station_E), ('F', station_F), ('G', station_G)]:
    _s = _f()
    register_station(_k, _s['name'], _s['t'], wse=_s['wse'], native_cadence_min=_s['native_cadence_min'], color=_s['color'])


def dropout_sweep(station, m=10, n_trials=800, seed=0):
    t_real = station['t']
    rng = np.random.default_rng(seed)
    kappa0, var0, _ = gram_and_edge(t_real, t_real, m=m)
    median_gap_h = np.median(np.diff(t_real)) / 60.0
    durations_h = [median_gap_h * k for k in (1, 2, 4, 8, 16, 32)]
    window_h = (t_real[-1] - t_real[0]) / 60.0
    durations_h = [d for d in durations_h if d <= 0.42 * window_h]
    edge_rows = []
    for d_h in durations_h:
        d_min = d_h * 60
        t_last = t_real[-1]
        keep = t_real[(t_real <= t_last - d_min) | (t_real == t_last)]
        if len(keep) < 15:
            continue
        kappa, var_edge, lev = gram_and_edge(keep, t_real, m=m)
        edge_rows.append((d_h, kappa, var_edge / var0, lev))
    kappas, inflations = ([], [])
    max_interior_h = durations_h[-1] if durations_h else median_gap_h * 8
    for _ in range(n_trials):
        d_min = rng.uniform(median_gap_h * 60, max_interior_h * 60)
        lo, hi = (t_real[1], t_real[-2] - d_min)
        start = rng.uniform(lo, hi) if hi > lo else lo
        mask = (t_real < start) | (t_real > start + d_min) | (t_real == t_real[0]) | (t_real == t_real[-1])
        grid = t_real[mask]
        if len(grid) < 15:
            continue
        kappa, var_edge, _ = gram_and_edge(grid, t_real, m=m)
        kappas.append(kappa)
        inflations.append(var_edge / var0)
    kappas = np.array(kappas)
    inflations = np.array(inflations)
    r2 = np.corrcoef(np.log10(kappas), inflations)[0, 1] ** 2 if len(kappas) > 2 else float('nan')
    return dict(kappa0=kappa0, var0=var0, edge_rows=edge_rows, kappas=kappas, inflations=inflations, r2=r2, median_gap_h=median_gap_h)

def _bootstrap_edge_var(x_nodes, weight_mult, noise_pool, m, trials, rng):
    w = trapezoid_weights(x_nodes) * weight_mult
    Phi = legendre_design(x_nodes, m)
    G = (Phi * w[:, None]).T @ Phi
    Ginv = np.linalg.pinv(G)
    nn = np.arange(m + 1)
    phi_edge = np.sqrt(nn + 0.5) * eval_legendre(nn, 1.0)
    coef_edge = Ginv @ phi_edge
    n = len(x_nodes)
    noise = rng.choice(noise_pool, size=(trials, n))
    fitted = noise @ (w[:, None] * Phi) @ coef_edge
    return np.var(fitted)

def redesign_comparison(station, m=10, trials=4000, seed=1, n_seeds=5):
    t_real = station['t']
    n = len(t_real)
    x_current = to_domain(t_real, t_real)
    noise_pool = np.diff(x_current)
    noise_pool = np.concatenate([noise_pool, -noise_pool]) * 0.05
    x_redesign = chebyshev_like_nodes(n)
    curr, redz = ([], [])
    for s in range(seed, seed + n_seeds):
        rng = np.random.default_rng(s)
        curr.append(_bootstrap_edge_var(x_current, np.ones_like(x_current), noise_pool, m, trials, rng))
        rng = np.random.default_rng(s + 1000)
        redz.append(_bootstrap_edge_var(x_redesign, np.ones_like(x_redesign), noise_pool, m, trials, rng))
    var_current, var_redesign = (float(np.mean(curr)), float(np.mean(redz)))
    pct_reduction = 100 * (1 - var_redesign / var_current)
    return dict(var_current=var_current, var_redesign=var_redesign, pct_reduction=pct_reduction)

def reweight_only_comparison(station, m=10, trials=4000, seed=1, n_seeds=5):
    t_real = station['t']
    x = to_domain(t_real, t_real)
    noise_pool = np.diff(x)
    noise_pool = np.concatenate([noise_pool, -noise_pool]) * 0.05
    cheb_w = chebyshev_weight(x)
    curr, rew = ([], [])
    for s in range(seed, seed + n_seeds):
        rng = np.random.default_rng(s)
        curr.append(_bootstrap_edge_var(x, np.ones_like(x), noise_pool, m, trials, rng))
        rng = np.random.default_rng(s + 2000)
        rew.append(_bootstrap_edge_var(x, cheb_w, noise_pool, m, trials, rng))
    var_current, var_reweight = (float(np.mean(curr)), float(np.mean(rew)))
    pct_reduction = 100 * (1 - var_reweight / var_current)
    return dict(var_current=var_current, var_reweight=var_reweight, pct_reduction=pct_reduction)

def real_wse_validation(station, m=10, trials=4000, seed=1):
    if station.get('wse') is None:
        return None
    t, wse = (station['t'], station['wse'])
    x = to_domain(t, t)
    Phi3 = legendre_design(x, 3)
    w3 = trapezoid_weights(x)
    G3 = (Phi3 * w3[:, None]).T @ Phi3
    b3 = (Phi3 * w3[:, None]).T @ wse
    coef3 = np.linalg.solve(G3, b3)
    resid = wse - Phi3 @ coef3
    resid_pool = np.concatenate([resid, -resid])
    rng = np.random.default_rng(seed)
    var_current = _bootstrap_edge_var(x, np.ones_like(x), resid_pool, m, trials, rng)
    x_redesign = chebyshev_like_nodes(len(x))
    var_redesign = _bootstrap_edge_var(x_redesign, np.ones_like(x_redesign), resid_pool, m, trials, rng)
    pct_reduction = 100 * (1 - var_redesign / var_current)
    return dict(resid_std=float(resid.std()), var_current=var_current, var_redesign=var_redesign, pct_reduction=pct_reduction)

def _Sm_legendre(x, m):
    nn = np.arange(m + 1)
    phi = np.sqrt(nn + 0.5)[:, None] * eval_legendre(nn[:, None], x[None, :])
    return (phi ** 2).sum(axis=0)

def _Sm_jacobi_endpoint(m, alpha, beta=None):
    if beta is None:
        beta = alpha
    from scipy.special import gammaln
    n = np.arange(m + 1)
    log_Pn1 = gammaln(n + alpha + 1) - gammaln(alpha + 1) - gammaln(n + 1)
    log_hn = (alpha + beta + 1) * np.log(2) - np.log(2 * n + alpha + beta + 1) + gammaln(n + alpha + 1) + gammaln(n + beta + 1) - gammaln(n + alpha + beta + 1) - gammaln(n + 1)
    log_phi2 = 2 * log_Pn1 - log_hn
    return np.exp(log_phi2).sum()

def fig_position_penalty_index(m_list=(5, 15, 40)):
    x = np.linspace(-0.999, 0.999, 2000)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m in m_list:
        Sm = _Sm_legendre(x, m)
        Sbar = (m + 1) / 2
        ax.plot(x, Sm / Sbar, label=f'm = {m}')
        ax.plot(1.0, m + 1, 'o', color='crimson')
    ax.set_xlabel('x')
    ax.set_ylabel('$\\Pi_m(x)$')
    ax.set_title('Position Penalty Index; dots mark $\\Pi_m(1)=m+1$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_position_penalty_index.pdf'))
    plt.close(fig)

def fig_envelope_and_ratio(m_list=(5, 15, 40), m_max=80):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    x = np.linspace(-0.999, 0.999, 2000)
    ax = axes[0]
    for m in m_list:
        ax.plot(x, _Sm_legendre(x, m) / m, label=f'm={m}')
    ax.plot(x, 1 / (np.pi * np.sqrt(1 - x ** 2)), 'k--', label='$1/(\\pi\\sqrt{1-x^2})$')
    ax.set_xlabel('x')
    ax.set_ylabel('$S_m(x)/m$')
    ax.set_ylim(0, 3)
    ax.set_title('Christoffel kernel vs. bulk envelope')
    ax.legend(fontsize=8)
    ax = axes[1]
    ms = np.arange(1, m_max + 1)
    ratio_exact = np.array([_Sm_legendre(np.array([1.0]), m)[0] / _Sm_legendre(np.array([0.0]), m)[0] for m in ms])
    ax.plot(ms, ratio_exact, '.', label='exact $S_m(1)/S_m(0)$')
    ax.plot(ms, np.pi / 2 * ms, 'k--', label='$(\\pi/2)m$')
    ax.set_xlabel('m')
    ax.set_ylabel('$S_m(1)/S_m(0)$')
    ax.set_title('Legendre endpoint-to-interior ratio')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_envelope_ratio.pdf'))
    plt.close(fig)

def fig_jacobi_classification(alphas=(-0.499, -0.25, 0.0, 0.5, 1.0), m_max=3000):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ms = np.unique(np.geomspace(50, m_max, 25).astype(int))
    for a in alphas:
        ratios = np.array([_Sm_jacobi_endpoint(m, a) / (m / np.pi) for m in ms])
        ax.loglog(ms, ratios, 'o-', label=f'$\\alpha=\\beta={a}$', ms=3)
    ax.set_xlabel('m')
    ax.set_ylabel('$S_m(1)/S_m(0)$')
    ax.set_title('Sharp classification: slope $=2\\alpha+1$')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_jacobi_classification.pdf'))
    plt.close(fig)

def fig_interior_classification(mus=(0.1, 0.35, 0.5, 1.0), m_max=12800):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ms = np.unique(np.geomspace(100, m_max, 20).astype(int))
    for mu in mus:
        a = mu - 0.5
        vals = np.array([_Sm_jacobi_endpoint(n, a) for n in ms // 2])
        ax.loglog(ms, vals, 'o-', ms=3, label=f'$\\mu={mu}$')
    ax.set_xlabel('m')
    ax.set_ylabel('$S_m(0;w_{\\mu,\\beta})$')
    ax.set_title('Interior classification: slope $=2\\mu+1$')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_interior_classification.pdf'))
    plt.close(fig)

def fig_weighted_montecarlo(m=10, n_nodes=40, trials=4000, seed=0):
    from numpy.polynomial.legendre import leggauss
    rng = np.random.default_rng(seed)
    x_gl, w_gl = leggauss(n_nodes)

    def var_ratio(alpha):
        w_extra = (1 - x_gl ** 2 + 0.05) ** alpha
        w = w_gl * w_extra
        Phi = legendre_design(x_gl, m)
        G = (Phi * w[:, None]).T @ Phi
        Ginv = np.linalg.pinv(G)
        nn = np.arange(m + 1)
        phi0 = np.sqrt(nn + 0.5) * eval_legendre(nn, 0.0)
        phi1 = np.sqrt(nn + 0.5) * eval_legendre(nn, 1.0)
        noise = rng.normal(size=(trials, n_nodes))
        coef0, coef1 = (Ginv @ phi0, Ginv @ phi1)
        f0 = noise @ (w[:, None] * Phi) @ coef0
        f1 = noise @ (w[:, None] * Phi) @ coef1
        return np.var(f1) / np.var(f0)
    ratios = [var_ratio(0.0), var_ratio(-0.3), var_ratio(-0.49)]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.bar(['Legendre\n(a=0)', 'a=-0.3', 'near-Chebyshev\n(a=-0.49)'], ratios, color=['#4C72B0', '#8B7BB0', '#B03A2E'])
    ax.set_ylabel('$\\widehat{\\mathrm{Var}}(\\hat p_{10}(1))/\\widehat{\\mathrm{Var}}(\\hat p_{10}(0))$')
    ax.set_title(f'Discrete weighted-fit variance ratio (m=10, {trials} trials)', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig5_weighted_montecarlo.pdf'))
    plt.close(fig)
    return ratios

def fig_jacobi_sweep(m=10, n_nodes=40, trials=3000, seed=0):
    from numpy.polynomial.legendre import leggauss
    rng = np.random.default_rng(seed)
    x_gl, w_gl = leggauss(n_nodes)
    alphas = np.linspace(-0.49, 1.0, 16)
    variances = []
    nn = np.arange(m + 1)
    phi1 = np.sqrt(nn + 0.5) * eval_legendre(nn, 1.0)
    for a in alphas:
        w = w_gl * (1 - x_gl ** 2 + 0.05) ** a
        Phi = legendre_design(x_gl, m)
        G = (Phi * w[:, None]).T @ Phi
        Ginv = np.linalg.pinv(G)
        coef1 = Ginv @ phi1
        noise = rng.normal(size=(trials, n_nodes))
        f1 = noise @ (w[:, None] * Phi) @ coef1
        variances.append(np.var(f1))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(alphas, variances, 'o-', color='#2E6FA7')
    ax.set_xlabel('Jacobi weight parameter ' + '$\\alpha$')
    ax.set_ylabel('Monte Carlo endpoint variance')
    ax.set_title(f'Endpoint variance across the Jacobi family (m={m})')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig6_jacobi_sweep.pdf'))
    plt.close(fig)

def generate_math_figures():
    print('Generating pure-math figures (Theorems 1-9, no station data)...')
    fig_position_penalty_index()
    fig_envelope_and_ratio()
    fig_jacobi_classification()
    fig_interior_classification()
    ratios = fig_weighted_montecarlo()
    print(f'    Fig.5 variance ratios (Legendre, a=-0.3, near-Chebyshev): {ratios[0]:.2f}, {ratios[1]:.2f}, {ratios[2]:.2f}')
    fig_jacobi_sweep()
    print('    -> figs 1-6 written to', OUT, '\n')

MAIN_STATIONS = ['A', 'B', 'C']

def fig_three_station_decoupling():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for key in MAIN_STATIONS:
        st = STATIONS[key]()
        sweep = dropout_sweep(st, m=10, n_trials=800, seed=0)
        edge_arr = np.array(sweep['edge_rows'])
        axes[0].semilogy(edge_arr[:, 0], edge_arr[:, 1], 'o-', color=st['color'], label=f'Station {key}')
        r2_val = sweep['r2']
        axes[1].scatter(sweep['kappas'], sweep['inflations'], s=6, alpha=0.35, color=st['color'], label=f'{key} ($R^2$={r2_val:.2f})')
    axes[0].set_xlabel('Dropout duration at reported edge (h)')
    axes[0].set_ylabel('$\\kappa(G)$')
    axes[0].set_title('Conditioning collapse, all 3 real stations')
    axes[0].legend(fontsize=8)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('$\\kappa(G)$ (random interior dropouts)')
    axes[1].set_ylabel('Var(edge) inflation')
    axes[1].set_title('Weakly correlated with ' + '$\\kappa(G)$' + ' on all 3')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig7_three_station_decoupling.pdf'))
    plt.close(fig)

def fig_three_station_redesign():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    keys = MAIN_STATIONS
    redesign_pcts, reweight_pcts = ([], [])
    for key in keys:
        st = STATIONS[key]()
        redesign_pcts.append(redesign_comparison(st)['pct_reduction'])
        reweight_pcts.append(reweight_only_comparison(st)['pct_reduction'])
    axes[0].bar(keys, redesign_pcts, color=[STATIONS[k]()['color'] for k in keys])
    axes[0].set_ylabel('% reduction in Var(edge)')
    axes[0].set_title('Redesigning sample placement\n(near-Chebyshev timing, same budget)')
    axes[1].bar(keys, reweight_pcts, color=[STATIONS[k]()['color'] for k in keys])
    axes[1].set_ylabel('% reduction in Var(edge)')
    axes[1].set_title('Reweighting an existing archive\n(no new samples)')
    for ax in axes:
        ax.axhline(0, color='k', lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig9_three_station_redesign.pdf'))
    plt.close(fig)
    return (dict(zip(keys, redesign_pcts)), dict(zip(keys, reweight_pcts)))

def appendix_station_summary():
    """Compact one-line-per-station table for the supplementary stations (D, E, F, G, ...),
    i.e. every registered station not in MAIN_STATIONS. Does not touch Table 1 or Figs 7/9."""
    extra_keys = [k for k in STATIONS.keys() if k not in MAIN_STATIONS]
    rows = []
    for key in extra_keys:
        st = STATIONS[key]()
        sweep = dropout_sweep(st, m=10, n_trials=800, seed=0)
        redesign = redesign_comparison(st)
        reweight = reweight_only_comparison(st)
        edge_kappa_max = max((r[1] for r in sweep['edge_rows']))
        rows.append(dict(
            key=key, name=st['name'], n=len(st['t']),
            cadence_min=st['native_cadence_min'],
            kappa0=sweep['kappa0'], kappa_max=edge_kappa_max,
            edge_var_infl=100 * (max((r[2] for r in sweep['edge_rows'])) - 1),
            r2=sweep['r2'],
            redesign_pct=redesign['pct_reduction'],
            reweight_pct=reweight['pct_reduction'],
        ))
    return rows

def generate_all():
    generate_math_figures()
    print(f'Generating figures/tables for all {len(STATIONS)} registered stations...\n')
    summary_lines = []
    for key, factory in STATIONS.items():
        station = factory()
        print(f"[{key}] {station['name']}  ({station['kind']}, n={len(station['t'])}, native cadence ~{station['native_cadence_min']:.0f} min)")
        sweep = dropout_sweep(station)
        redesign = redesign_comparison(station)
        reweight = reweight_only_comparison(station)
        wse_check = real_wse_validation(station)
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
        ax = axes[0]
        ax2 = ax.twinx()
        edge_arr = np.array(sweep['edge_rows'])
        ax.semilogy(edge_arr[:, 0], edge_arr[:, 1], 'o-', color='#B03A2E')
        ax2.plot(edge_arr[:, 0], edge_arr[:, 2], 's-', color='#2E6FA7')
        ax.set_xlabel('Dropout duration at reported edge (h)')
        ax.set_ylabel('$\\kappa(G)$', color='#B03A2E')
        ax2.set_ylabel('Var(edge) inflation', color='#2E6FA7')
        ax.set_title(f"{key}: conditioning vs. edge variance\n(native cadence {station['native_cadence_min']:.0f} min)")
        ax = axes[1]
        ax.scatter(sweep['kappas'], sweep['inflations'], s=8, alpha=0.3, color='#3C7A54')
        ax.set_xscale('log')
        ax.set_xlabel('$\\kappa(G)$ (n=%d random interior dropouts)' % len(sweep['kappas']))
        ax.set_ylabel('Var(edge) inflation')
        ax.set_title(f"$R^2$={sweep['r2']:.2f}")
        fig.tight_layout()
        fname = os.path.join(OUT, f'station_{key}_dropout.pdf')
        fig.savefig(fname)
        plt.close(fig)
        fig2, ax = plt.subplots(figsize=(5.4, 3.8))
        labels = ['Current\n(real, uniform-nominal)', 'Reweighted\n(same archive)', 'Redesigned\n(near-Chebyshev)']
        values = [redesign['var_current'], reweight['var_reweight'], redesign['var_redesign']]
        bars = ax.bar(labels, values, color=['#4C72B0', '#B7950B', '#B03A2E'])
        ax.set_ylabel('Var(edge), bootstrapped noise')
        ax.set_title(f"{key}: redesign cuts {redesign['pct_reduction']:.1f}%, reweight-only {reweight['pct_reduction']:.1f}%", fontsize=9)
        fig2.tight_layout()
        fname2 = os.path.join(OUT, f'station_{key}_redesign.pdf')
        fig2.savefig(fname2)
        plt.close(fig2)
        edge_kappa_max = max((r[1] for r in sweep['edge_rows']))
        print(f"    kappa0={sweep['kappa0']:.3f}  ->  max kappa at reported edge={edge_kappa_max:.3e}  ({np.log10(edge_kappa_max / sweep['kappa0']):.1f} orders of magnitude)")
        print(f"    edge-variance inflation saturates at +{100 * (max((r[2] for r in sweep['edge_rows'])) - 1):.1f}%   R^2={sweep['r2']:.3f}")
        print(f"    redesign (near-Chebyshev):  {redesign['pct_reduction']:.1f}% reduction")
        print(f"    reweight-only (same archive): {reweight['pct_reduction']:.1f}% reduction")
        if wse_check:
            print(f"    independent real-WSE-residual check: {wse_check['pct_reduction']:.1f}% reduction (resid std={wse_check['resid_std']:.4f} m)")
        check = reliability_check(station['t'], m=10, kappa_warn=10 * sweep['kappa0'], kappa_critical=10000.0 * sweep['kappa0'])
        print(f"    ReliabilityChecker: kappa={check['kappa']:.3g}, status={check['status']}")
        print(f'    -> {fname}\n    -> {fname2}\n')
        summary_lines.append(f"{key} ({station['kind']}, n={len(station['t'])}): kappa0={sweep['kappa0']:.2f} -> {sweep['kappas'].max():.2e}, R2={sweep['r2']:.2f}, redesign={redesign['pct_reduction']:.1f}%, reweight={reweight['pct_reduction']:.1f}%, ReliabilityChecker={check['status']}")
    fig_three_station_decoupling()
    redz, rewt = fig_three_station_redesign()
    print('Combined 3-station figures written: fig7_three_station_decoupling.pdf, fig9_three_station_redesign.pdf')
    print('  redesign %:', {k: round(v, 1) for k, v in redz.items()})
    print('  reweight %:', {k: round(v, 1) for k, v in rewt.items()})
    print('\nAppendix: supplementary stations (not in main Table 1 / Figs 7,9)...')
    appendix_rows = appendix_station_summary()
    for r in appendix_rows:
        print(f"  [{r['key']}] {r['name']}: n={r['n']}, kappa0={r['kappa0']:.2f} -> {r['kappa_max']:.3e} "
              f"({np.log10(r['kappa_max']/r['kappa0']):.1f} ord.), edge-var infl +{r['edge_var_infl']:.1f}%, "
              f"R2={r['r2']:.2f}, redesign={r['redesign_pct']:.1f}%, reweight={r['reweight_pct']:.1f}%")
    all_redesign_pcts = list(redz.values()) + [r['redesign_pct'] for r in appendix_rows]
    n_total_stations = len(redz) + len(appendix_rows)
    avg_redesign = sum(all_redesign_pcts) / len(all_redesign_pcts)
    print(f'\nAll stations tested so far: {n_total_stations}. Average redesign reduction across all: {avg_redesign:.1f}%')
    with open(os.path.join(OUT, 'APPENDIX_SUMMARY.txt'), 'w') as f:
        for r in appendix_rows:
            f.write(f"{r['key']} ({r['name']}, n={r['n']}, cadence~{r['cadence_min']:.0f} min): "
                    f"kappa0={r['kappa0']:.2f} -> {r['kappa_max']:.3e}, R2={r['r2']:.2f}, "
                    f"redesign={r['redesign_pct']:.1f}%, reweight={r['reweight_pct']:.1f}%\n")
    print('\nIllustrative per-station economic benefit (Sec. "Economic Impact"):')
    print('  Framing: at a fixed sensor budget, a redesign that cuts Var(edge) by p% delivers the')
    print('  same reliability a station would otherwise need 1/(1-p) times its current budget to reach')
    print('  by adding co-located hardware -- i.e. redesign is worth (1/(1-p) - 1) "equivalent extra gauges", free.')
    cost_per_gauge_usd = 15000
    econ = economic_benefit(redz['C'], n_sensors=1, cost_per_sensor=cost_per_gauge_usd)
    print(f"  Ambarampalayam-profile station, redesign at {redz['C']:.1f}%: worth ~{econ['sensors_avoided']:.2f} equivalent extra co-located gauges")
    print(f"    (~${econ['savings']:,.0f} at an illustrative ${cost_per_gauge_usd:,}/gauge proxy, for zero marginal hardware)")
    print('  This is a per-station multiplier, not a network total: real per-station cost and cadence vary')
    print('  across CWC/CPCB stations, so a network-wide rupee figure would need CWC-internal cost data.')
    with open(os.path.join(OUT, 'SUMMARY.txt'), 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print('Done. See', OUT)

def launch_dashboard():
    fig = plt.figure(figsize=(11.5, 7.2))
    fig.suptitle('The Christoffel Position Penalty -- live explorer', fontsize=13)
    ax_edge = fig.add_axes([0.08, 0.42, 0.38, 0.44])
    ax_scatter = fig.add_axes([0.55, 0.42, 0.38, 0.44])
    ax_bar = fig.add_axes([0.08, 0.08, 0.24, 0.26])
    ax_radio = fig.add_axes([0.38, 0.06, 0.12, 0.28])
    ax_m = fig.add_axes([0.58, 0.3, 0.32, 0.03])
    ax_trials = fig.add_axes([0.58, 0.24, 0.32, 0.03])
    ax_status = fig.add_axes([0.58, 0.05, 0.35, 0.17])
    ax_status.axis('off')
    state = {'key': 'A', 'm': 10, 'trials': 400}

    def redraw(_=None):
        for a in (ax_edge, ax_scatter, ax_bar):
            a.clear()
        ax_status.clear()
        ax_status.axis('off')
        station = STATIONS[state['key']]()
        sweep = dropout_sweep(station, m=state['m'], n_trials=state['trials'])
        redesign = redesign_comparison(station, m=state['m'], trials=max(500, state['trials']), n_seeds=1)
        reweight = reweight_only_comparison(station, m=state['m'], trials=max(500, state['trials']), n_seeds=1)
        edge_arr = np.array(sweep['edge_rows'])
        ax_edge.semilogy(edge_arr[:, 0], edge_arr[:, 1], 'o-', color='#B03A2E')
        ax_edge2 = ax_edge.twinx()
        ax_edge2.plot(edge_arr[:, 0], edge_arr[:, 2], 's-', color='#2E6FA7')
        ax_edge.set_xlabel('Dropout duration at edge (h)')
        ax_edge.set_ylabel('$\\kappa(G)$', color='#B03A2E')
        ax_edge2.set_ylabel('Var(edge) inflation', color='#2E6FA7')
        ax_edge.set_title('Conditioning vs. edge-variance', fontsize=10)
        ax_scatter.scatter(sweep['kappas'], sweep['inflations'], s=6, alpha=0.3, color='#3C7A54')
        ax_scatter.set_xscale('log')
        ax_scatter.set_xlabel('$\\kappa(G)$')
        ax_scatter.set_ylabel('Var(edge) inflation')
        ax_scatter.set_title(f"$R^2$={sweep['r2']:.2f}  (n={len(sweep['kappas'])})", fontsize=10)
        ax_bar.bar(['Current', 'Reweight', 'Redesign'], [redesign['var_current'], reweight['var_reweight'], redesign['var_redesign']], color=['#4C72B0', '#B7950B', '#B03A2E'])
        ax_bar.set_title(f"redesign -{redesign['pct_reduction']:.0f}%, reweight -{reweight['pct_reduction']:.0f}%", fontsize=9)
        ax_bar.tick_params(axis='x', labelsize=8)
        ax_status.text(0.0, 0.85, station['name'], fontsize=10, weight='bold', wrap=True)
        ax_status.text(0.0, 0.6, f"n = {len(station['t'])} readings, degree m = {state['m']}", fontsize=9)
        ax_status.text(0.0, 0.42, f"native cadence ~ {sweep['median_gap_h'] * 60:.0f} min", fontsize=9)
        ax_status.text(0.0, 0.24, f"kappa(G): {sweep['kappa0']:.2f} -> {sweep['kappas'].max():.2e}", fontsize=9)
        ax_status.text(0.0, 0.06, f"edge-var inflation saturates at +{100 * (max((r[2] for r in sweep['edge_rows'])) - 1):.1f}%", fontsize=9)
        fig.canvas.draw_idle()
    radio = RadioButtons(ax_radio, ('A', 'B', 'C'))

    def on_station(label):
        state['key'] = label
        redraw()
    radio.on_clicked(on_station)
    s_m = Slider(ax_m, 'degree m', 2, 20, valinit=state['m'], valstep=1)

    def on_m(val):
        state['m'] = int(val)
        redraw()
    s_m.on_changed(on_m)
    s_trials = Slider(ax_trials, 'trials', 50, 800, valinit=state['trials'], valstep=50)

    def on_trials(val):
        state['trials'] = int(val)
        redraw()
    s_trials.on_changed(on_trials)
    redraw()
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generate', action='store_true', help='Batch-generate every figure/table and exit (no display needed).')
    args = parser.parse_args()
    if args.generate:
        matplotlib.use('Agg')
        generate_all()
    else:
        launch_dashboard()
