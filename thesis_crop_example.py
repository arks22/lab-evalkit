import numpy as np
import cv2
import os
from sunpy.map import Map
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import astropy.units as u

# フォントスタイルをセット
rc('font', family='serif')
rc('font', serif='DejaVu Serif')
rc('text', usetex=False)

# フォントサイズを設定
matplotlib.rcParams.update({'font.size': 13})

sample = 'home/sasaki/evaluate_factory/aia211/202311120927_testof_111136/pd_fits/01/aia.lev1_euv_12s.2022-02-20T235959Z.211.image_lev1.fits'
smap = Map(sample)
