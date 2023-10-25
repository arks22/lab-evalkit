import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import imageio
from skimage.metrics import structural_similarity
import json

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

from astropy.io import fits
import astropy.units as u
from astropy.time import TimeDelta
from astropy.coordinates import SkyCoord

from sunpy.map import Map
from sunpy.physics.differential_rotation import differential_rotate
from sunpy.coordinates import RotatedSunFrame
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst

from evaluate_utils import rotate_map
from evaluate_utils import resize_map
from evaluate_utils import mask_map
from evaluate_utils import calculate_metrics
from evaluate_utils import calculate_error
from evaluate_utils import calculate_ssim
from evaluate_utils import find_max_in_ndarrays
from evaluate_utils import fig_to_ndarray
from evaluate_utils import get_fits_img
from evaluate_utils import plot_map
from evaluate_utils import plot_histogram
from evaluate_utils import plot_metrics
from evaluate_utils import plot_diff

def evaluate_activeregion(timestamp, gt_dirs, pd_dirs, len_test, len_seq, len_output, len_input, img_shape):
    gt_mean      = np.full((len_test, len_seq), np.nan)
    pd_mean      = np.full_like(gt_mean, np.nan)
    sp_mean      = np.full_like(gt_mean, np.nan)

    error_gt_pd  = np.full_like(gt_mean, np.nan)
    error_gt_sp  = np.full_like(gt_mean, np.nan)
    ssim_gt_pd   = np.full_like(gt_mean, np.nan)
    ssim_gt_sp   = np.full_like(gt_mean, np.nan)

    with open('active_region.json', 'r') as file:
        init_rectangle_coords = json.load(file)
    
    for i in tqdm(range(len_test), leave=False):
        gt_fits_files = sorted(glob(f'{gt_dirs[i]}/*'))
        pd_fits_files = sorted(glob(f'{pd_dirs[i]}/*'))

        # --------------------------------------------------------------------#
        # マップを作成
        gt_maps, pd_maps, sp_maps, = [], [], []
        gt_sub_maps, pd_sub_maps, sp_sub_maps = [], [], []
        diff_gt_pds, diff_gt_sps = [], []

        # バウンディングボックスの初期位置  
        init_map = resize_map(Map(gt_fits_files[0]), img_shape)
        init_bottom_left = SkyCoord(init_rectangle_coords[i][0][0]*u.arcsec, init_rectangle_coords[i][0][1]*u.arcsec, frame=init_map.coordinate_frame)
        init_top_right   = SkyCoord(init_rectangle_coords[i][1][0]*u.arcsec, init_rectangle_coords[i][1][1]*u.arcsec, frame=init_map.coordinate_frame)

        bbox_coords = []
        for t in tqdm(range(len_seq), leave=False, desc='Map Generation: '):
            gt_map = Map(gt_fits_files[t])
            gt_map = resize_map(gt_map, img_shape)

            dummy_map = Map(np.full(img_shape, 0), gt_map.meta)

            if t < len_input:
                pd_map = dummy_map
                sp_map = dummy_map
            else:
                sp_map = Map(rotate_map(gt_maps[len_input-1], (t-(len_input-1)) * 4).data, gt_map.meta) #rotate_mapの計算によるmetaデータの変更を阻止
                pd_map = Map(Map(pd_fits_files[t - len_input]).data, gt_map.meta)

                gt_map = mask_map(gt_map, sp_map)
                pd_map = mask_map(pd_map, sp_map)

            rotated_bottom_left = SkyCoord(RotatedSunFrame(base=init_bottom_left, rotated_time=gt_map.date)).transform_to(gt_map.coordinate_frame)
            rotated_top_right   = SkyCoord(RotatedSunFrame(base=init_top_right,   rotated_time=gt_map.date)).transform_to(gt_map.coordinate_frame)
            bbox_coords.append((rotated_bottom_left, rotated_top_right))

            gt_sub_map = gt_map.submap(rotated_bottom_left, top_right=rotated_top_right)
            pd_sub_map = pd_map.submap(rotated_bottom_left, top_right=rotated_top_right)
            sp_sub_map = sp_map.submap(rotated_bottom_left, top_right=rotated_top_right)

            diff_gt_pd_raw = pd_sub_map.data - gt_sub_map.data
            diff_gt_sp_raw = sp_sub_map.data - gt_sub_map.data
            diff_gt_pd = np.sign(diff_gt_pd_raw) * np.log1p(np.abs(diff_gt_pd_raw))
            diff_gt_sp = np.sign(diff_gt_sp_raw) * np.log1p(np.abs(diff_gt_sp_raw))

            gt_maps.append(gt_map)
            pd_maps.append(pd_map)
            sp_maps.append(sp_map)
            gt_sub_maps.append(gt_sub_map)
            pd_sub_maps.append(pd_sub_map)
            sp_sub_maps.append(sp_sub_map)
            diff_gt_pds.append(diff_gt_pd)
            diff_gt_sps.append(diff_gt_sp)
        

        # --------------------------------------------------------------------#
        # グラフの値域を決めるため、メトリクスを先に計算
        for t in tqdm(range(len_seq), leave=False, desc='Metrics Calculation: '):
            gt_img = gt_sub_maps[t].data.astype(np.float32)
            pd_img = pd_sub_maps[t].data
            sp_img = sp_sub_maps[t].data

            gt_mean[i, t], _ = calculate_metrics(gt_img)
            if t >= len_input:
                pd_mean[i, t], _     = calculate_metrics(pd_img)
                sp_mean[i, t], _     = calculate_metrics(sp_img)

                error_gt_pd[i, t]     = calculate_error(gt_mean[i, t], pd_mean[i, t])
                error_gt_sp[i, t]     = calculate_error(gt_mean[i, t], sp_mean[i, t])
                ssim_gt_pd[i,t]       = calculate_ssim(gt_img, pd_img)
                ssim_gt_sp[i,t]       = calculate_ssim(gt_img, sp_img)

        # --------------------------------------------------------------------#
        # プロット
        save_dir = f'./{timestamp}/activeregion/'
        os.makedirs(save_dir, exist_ok=True)
        writer = imageio.get_writer(f'{save_dir}/{i}.mp4', fps=5)

        for t in tqdm(range(len_seq), leave=False, desc='Plot: '):
            fig = plt.figure(figsize=(24,32), tight_layout=True)
            gspec = gridspec.GridSpec(14, 3) #(縦、横)
            plt.rcParams["font.size"] = 14

            plot_map(fig, gspec[0:3,0], gt_maps[t], draw_bbox=True, bottom_left=bbox_coords[t][0], top_right=bbox_coords[t][1])
            plot_map(fig, gspec[0:3,1], pd_maps[t], draw_bbox=True, bottom_left=bbox_coords[t][0], top_right=bbox_coords[t][1])
            plot_map(fig, gspec[0:3,2], sp_maps[t], draw_bbox=True, bottom_left=bbox_coords[t][0], top_right=bbox_coords[t][1])

            plot_map(fig, gspec[3:6,0], gt_sub_maps[t])
            plot_map(fig, gspec[3:6,1], pd_sub_maps[t])
            plot_map(fig, gspec[3:6,2], sp_sub_maps[t])

            if t >= len_input:
                diff_max = max(find_max_in_ndarrays(diff_gt_pds), find_max_in_ndarrays(diff_gt_sps))
                plot_diff(fig, gspec[6:9,1], gspec[9,1], diff_gt_pds[t], diff_max)
                plot_diff(fig, gspec[6:9,2], gspec[9,2], diff_gt_sps[t], diff_max)

            intensity_list = [('GT', gt_mean[i], '#1f77b4'), ('Prediction', pd_mean[i], '#ff7f0e'), ('Sunpy', sp_mean[i], '#2ca02c')]
            mean_min = min(np.nanmin(gt_mean[i]), np.nanmin(pd_mean[i]), np.nanmin(sp_mean[i])) * 0.95
            mean_max = max(np.nanmax(gt_mean[i]), np.nanmax(pd_mean[i]), np.nanmax(sp_mean[i])) * 1.05
            plot_metrics(fig, gspec[10:12,0:], intensity_list, 'Mean Intensity', 'Intensity', mean_min, mean_max, 0, len_seq, len_input, is_move=True, t=t)

            error_list = [('GT - Prediction', error_gt_pd[i], '#ff7f0e'), ('GT - Sunpy', error_gt_sp[i], '#2ca02c')]
            plot_metrics(fig, gspec[12:14,0:], error_list, 'Error of Mean Intensity', '%',-30, 30, 0, len_seq, len_input, is_move=True, t=t, is_error=True)

            """
            ssim_list = [('GT - Prediction', ssim_gt_pd[i], '#ff7f0e'), ('GT - Sunpy', ssim_gt_sp[i], '#2ca02c')]
            plot_metrics(fig, gspec[6,0:], ssim_list, 'Average SSIM', 'SSIM',  0.5, 1, 0, len_seq, len_input, is_move=True, t=t)
            """

            writer.append_data(fig_to_ndarray(fig))
            plt.close(fig)
            plt.clf()

        writer.close()

        # まとめ
        fig = plt.figure(figsize=(20,15), tight_layout=True)
        gspec = gridspec.GridSpec(2, 1)
        plt.rcParams["font.size"] = 12

        mean_error_gt_pd = np.nanmean(np.abs(error_gt_pd), axis=0)
        mean_error_gt_sp = np.nanmean(np.abs(error_gt_sp), axis=0)
        mean_ssim_gt_pd  = np.nanmean(ssim_gt_pd, axis=0)
        mean_ssim_gt_sp  = np.nanmean(ssim_gt_sp, axis=0)

        error_list = [('GT - Prediction', mean_error_gt_pd, '#ff7f0e'), ('GT - Sunpy', mean_error_gt_sp, '#2ca02c')]
        plot_metrics(fig, gspec[0], error_list, 'Average Absolute Error of Mean Intensity', '%',0, 20, len_input, len_seq, len_input, is_error=True, is_write_val=True)

        ssim_list = [('GT - Prediction', mean_ssim_gt_pd, '#ff7f0e'), ('GT - Sunpy', mean_ssim_gt_sp, '#2ca02c')]
        plot_metrics(fig, gspec[1], ssim_list, 'Average SSIM', 'SSIM',  0.55, 1, len_input, len_seq, len_input, is_write_val=True)

        fig.savefig(f'./{timestamp}/summary_active_region.png')
