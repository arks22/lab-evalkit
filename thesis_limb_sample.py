import numpy as np
import cv2
import os
from sunpy.map import Map
from astropy.coordinates import SkyCoord
import glob
from tqdm import tqdm

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

#dir = 'aia211/202311121626_testof_120336'
dir = 'aia3wave/202311121715'
out_dir = 'thesis_exp2'

sample_dirs_gt = sorted(glob.glob(os.path.join(dir, 'gt_fits', '*')))
sample_dirs_pd = sorted(glob.glob(os.path.join(dir, 'pd_fits', '*')))

sample_i = [ 3, 12 ]
sample_t_gt = [11, 12, 17, 23]
sample_t_pd = [0, 5, 11]

def resize_map(smap, new_shape):
    # astropy Quantityオブジェクトに新しいシェイプを変換
    new_dimensions = u.Quantity(new_shape, u.pixel)

    # データをリサイズ
    resampled_map = smap.resample(new_dimensions)

    return resampled_map


def plot_submap(smap, ax_index,t):  
    bottom_left = SkyCoord(-1100* u.arcsec, -550* u.arcsec, frame=smap.coordinate_frame)
    top_right   = SkyCoord(-600 * u.arcsec, 550 * u.arcsec, frame=smap.coordinate_frame)

    submap = smap.submap(bottom_left, top_right=top_right)
    ax = fig.add_subplot(2, 4, ax_index+1, projection=submap)
    submap.plot(clip_interval=(1,99.99)*u.percent)
    
    ax.set_title(f't = {t} ({(t)*4} h)')
    ax.coords[0].set_ticks_visible(False)  # X軸の目盛りを無効化
    ax.coords[1].set_ticks_visible(False)  # Y軸の目盛りを無効化
    ax.coords[0].set_ticklabel_visible(False)  # X軸のラベルを無効化
    ax.coords[1].set_ticklabel_visible(False)  # X軸のラベルを無効化

for i in tqdm(range(len(sample_dirs_gt))):

    if i in sample_i: # exmapleに該当するディレクトリのみ
        gt_files = sorted(glob.glob(os.path.join(sample_dirs_gt[i], '*.fits')))
        pd_files = sorted(glob.glob(os.path.join(sample_dirs_pd[i], '*.fits')))
        
        gt_maps = [Map(path) for path in gt_files]
        pd_maps = [Map(path) for path in pd_files]
        
        fig = plt.figure(figsize=(6, 6))
        
        ax_index = 0
        for t, gt_map in enumerate(gt_maps):
            if t in sample_t_gt:
                gt_map = resize_map(gt_map, (512, 512))
                plot_submap(gt_map,ax_index, t)
                ax_index += 1
        
        ax_index += 1 # 左下に空白ax

        for t, pd_map in enumerate(pd_maps):
            if t in sample_t_pd:
                resize_gt_meta = resize_map(gt_maps[t], (512,512)).meta
                pd_map = Map(pd_map.data, resize_gt_meta)
                plot_submap(pd_map, ax_index, t+12)
                ax_index += 1
        
        plt.tight_layout()
        plt.savefig(f'{out_dir}/limb_sample_{i}.png')
        plt.close()
    
    

