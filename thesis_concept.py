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
rc('font', serif='Times New Roman')
rc('text', usetex=False)

# フォントサイズを設定
matplotlib.rcParams.update({'font.size': 13})

dir = 'concept_exp2/193'

gt_images = sorted(glob.glob(os.path.join(dir, 'gt', '*.fits')))
pd_images = sorted(glob.glob(os.path.join(dir, 'pd', '*.fits')))

gt = [Map(path) for path in gt_images]
pd = [Map(path) for path in pd_images]

def plot_map(smap, index, prefix):
    fig = plt.figure(figsize=(4, 4),tight_layout=True)

    # subplotを追加
    ax = fig.add_subplot(1, 1, 1, projection=smap)
    smap.plot(clip_interval=(1, 99.99)*u.percent)
    ax.set_title('')
    ax.coords[0].set_ticks_visible(False)  # X軸の目盛りを無効化
    ax.coords[1].set_ticks_visible(False)  # Y軸の目盛りを無効化
    ax.coords[0].set_ticklabel_visible(False)  # X軸のラベルを無効化
    ax.coords[1].set_ticklabel_visible(False)  # Y軸のラベルを無効化
    ax.grid(False)

    # 図を保存
    plt.savefig(f'concept_exp2/193/{prefix}_{index}.png')
    # figureを閉じる
    plt.close(fig)
    plt.clf()

# gtの各要素に対してループを実行
for index, smap in enumerate(gt):
    plot_map(smap, index, 'gt')

for index, smap in enumerate(pd):
    plot_map(smap, index+12, 'pd')
