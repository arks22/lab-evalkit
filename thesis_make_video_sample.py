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

#dir = 'aia3wave/202311121715'
dir = 'aia211/202311121626_testof_120336'
out_dir = 'thesis_exp1'

gt_images = sorted(glob.glob(os.path.join(dir, 'gt_fits/01', '*.fits')))
pd_images = sorted(glob.glob(os.path.join(dir, 'pd_fits/01', '*.fits')))

gt = [Map(path) for path in gt_images]
pd = [Map(path) for path in pd_images]

# gt
fig = plt.figure(figsize=(12, 18))
index = 0

for j in range(4):
    for i in range(6):
        smap = gt[index]
        ax = fig.add_subplot(6, 4, index+1, projection=smap)
        smap.plot(clip_interval=(1,99.99)*u.percent)
        ax.set_title(f't = {index} ({index*4} h)')
        ax.coords[0].set_ticks_visible(False)  # X軸の目盛りを無効化
        ax.coords[1].set_ticks_visible(False)  # Y軸の目盛りを無効化
        ax.coords[0].set_ticklabel_visible(False)  # X軸のラベルを無効化
        ax.coords[1].set_ticklabel_visible(False)  # X軸のラベルを無効化
        ax.grid(False)
        index += 1

plt.tight_layout()
plt.savefig(f'{out_dir}/gt.png')
plt.close()

# pd
fig = plt.figure(figsize=(12, 9))
index = 0

for j in range(4):
    for i in range(3):
        smap = pd[index]
        ax = fig.add_subplot(3, 4, index+1, projection=smap)
        smap.plot(clip_interval=(1,99.99)*u.percent)
        ax.set_title(f't = {12 + index} ({(12+index)*4} h)')
        ax.coords[0].set_ticks_visible(False)  # X軸の目盛りを無効化
        ax.coords[1].set_ticks_visible(False)  # Y軸の目盛りを無効化
        ax.coords[0].set_ticklabel_visible(False)  # X軸のラベルを無効化
        ax.coords[1].set_ticklabel_visible(False)  # X軸のラベルを無効化
        ax.grid(False)
        index += 1

plt.tight_layout()
plt.savefig(f'{out_dir}/pd.png')
plt.close()