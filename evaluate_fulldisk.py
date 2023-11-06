import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import imageio
from skimage.metrics import structural_similarity

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

from astropy.io import fits
import astropy.units as u
from astropy.time import TimeDelta
from astropy.coordinates import SkyCoord

from sunpy.map import Map
from sunpy.physics.differential_rotation import differential_rotate
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
from evaluate_utils import plot_images
from evaluate_utils import plot_map
from evaluate_utils import plot_histogram
from evaluate_utils import plot_scatter
from evaluate_utils import plot_metrics
from evaluate_utils import plot_diff
from evaluate_utils import find_limb_deg
from evaluate_utils import generate_submap_by_lng
from evaluate_utils import generate_mean_map_by_lng
from evaluate_utils import generate_diff_map
from evaluate_utils import calculate_blur_with_fft


def evaluate_fulldisk(dst_dir, gt_dirs, pd_dirs, len_test, len_seq, len_output, len_input, args):
    gt_mean      = np.full((len_test, len_seq), np.nan)
    pd_mean      = np.full_like(gt_mean, np.nan)
    dr_mean      = np.full_like(gt_mean, np.nan)
    gt_crop_mean  = np.full_like(gt_mean, np.nan)
    pd_crop_mean  = np.full_like(gt_mean, np.nan)

    num_regions = 5
    gt_pd_error_lng = np.full((len_test, len_seq, num_regions), np.nan)
    gt_dr_error_lng = np.full((len_test, len_seq, num_regions), np.nan)

    gt_east_limb_last_input  = np.full((len_test), np.nan)
    gt_east_limb_last_output = np.full((len_test), np.nan)
    pd_east_limb_last_output = np.full((len_test), np.nan)

    error_gt_pd       = np.full_like(gt_mean, np.nan)
    error_crop_gt_dr  = np.full_like(gt_mean, np.nan)
    error_crop_gt_pd  = np.full_like(gt_mean, np.nan)

    ssim_gt_pd        = np.full_like(gt_mean, np.nan)
    ssim_crop_gt_dr   = np.full_like(gt_mean, np.nan)
    ssim_crop_gt_pd   = np.full_like(gt_mean, np.nan)
    
    blur_fft = np.full_like(gt_mean, np.nan)

    img_shape = (args.img_size, args.img_size)

    # テストセットの数だけループ
    for i in tqdm(range(len_test), leave=False):
        gt_fits_files = sorted(glob(f'{gt_dirs[i]}/*'))
        pd_fits_files = sorted(glob(f'{pd_dirs[i]}/*'))

        # --------------------------------------------------------------------#
        # マップを生成
        gt_maps, pd_maps, dr_maps, gt_crop_maps, pd_crop_maps = [], [], [], [], [] 
        diff_gt_pds, diff_gt_drs = [], []
        diff_lng_gt_pds, diff_lng_gt_drs = [], []
        submaps = []

        for t in tqdm(range(len_seq), leave=False, desc='Map Generation: '):
            gt_map = Map(gt_fits_files[t])
            gt_map = resize_map(gt_map, img_shape)

            dummy_map = Map(np.full(img_shape, 0), gt_map.meta)

            if t < len_input:
                pd_map     = dummy_map
                dr_map     = dummy_map
                gt_crop_map = dummy_map
                pd_crop_map = dummy_map
                diff_gt_pd = dummy_map.data
                diff_gt_dr = dummy_map.data
                diff_lng_gt_pd = dummy_map.data
                diff_lng_gt_dr = dummy_map.data

                # 最終入力で東のリムの平均輝度を計算
                if t == len_input-1:
                    _, gt_mean_lng_limb = generate_mean_map_by_lng(gt_map, num_regions)
                    gt_east_limb_last_input[i] = gt_mean_lng_limb[num_regions-1] #最も東の領域

            else:
                # PDマップを読み込み
                pd_map = Map(Map(pd_fits_files[t - len_input]).data, gt_map.meta)

                # 差動回転マップを生成
                dr_map = rotate_map(gt_maps[len_input-1], (t-(len_input-1)) * 4)
                dr_map = Map(dr_map.data, gt_map.meta)

                # 差動回転マップは円盤の外側がNaNのため、同じ条件で評価するために差動回転マップの形に合わせてマップをクロップ
                gt_crop_map = mask_map(gt_map, dr_map)
                pd_crop_map = mask_map(pd_map, dr_map)

                # ピクセルごとの差分マップデータ
                diff_gt_pd = generate_diff_map(gt_crop_map, pd_crop_map).data
                diff_gt_dr = generate_diff_map(gt_crop_map, dr_map).data
    
                #経度ごとの平均輝度とそのマップ
                mean_map_gt, gt_mean_lng = generate_mean_map_by_lng(gt_crop_map, num_regions)
                mean_map_pd, pd_mean_lng = generate_mean_map_by_lng(pd_crop_map, num_regions)
                mean_map_dr, dr_mean_lng = generate_mean_map_by_lng(dr_map,     num_regions)

                # 経度ごとの平均輝度の誤差を計算
                gt_pd_error_lng[i, t] = calculate_error(pd_mean_lng, gt_mean_lng)
                gt_dr_error_lng[i, t] = calculate_error(dr_mean_lng, gt_mean_lng)

                # 経度ごとの平均輝度のマップから誤差マップを生成
                diff_lng_gt_pd = generate_diff_map(mean_map_gt, mean_map_pd).data
                diff_lng_gt_dr = generate_diff_map(mean_map_gt, mean_map_dr).data

                # 最終出力で東のリムの平均輝度を計算
                if t == len_seq-1:
                    _, gt_mean_lng_limb = generate_mean_map_by_lng(gt_map, num_regions)
                    _, pd_mean_lng_limb = generate_mean_map_by_lng(pd_map, num_regions)
                    gt_east_limb_last_output[i] = gt_mean_lng_limb[num_regions-1] #最も東の領域
                    pd_east_limb_last_output[i] = pd_mean_lng_limb[num_regions-1]

            # 各マップをリストに追加
            gt_maps.append(gt_map)
            pd_maps.append(pd_map)
            dr_maps.append(dr_map)
            gt_crop_maps.append(gt_crop_map)
            pd_crop_maps.append(pd_crop_map)
            diff_gt_pds.append(diff_gt_pd)
            diff_gt_drs.append(diff_gt_dr)
            diff_lng_gt_pds.append(diff_lng_gt_pd)
            diff_lng_gt_drs.append(diff_lng_gt_dr)

        # --------------------------------------------------------------------#
        # グラフの値域を決めるため、メトリクスを先に計算
        for t in tqdm(range(len_seq), leave=False, desc='Metrics Calculation: '):
            gt_img = gt_maps[t].data.astype(np.float32)
            pd_img = pd_maps[t].data
            dr_img = dr_maps[t].data
            gt_crop_img = gt_crop_maps[t].data
            pd_crop_img = pd_crop_maps[t].data

            gt_mean[i, t], _ = calculate_metrics(gt_img)
            
            if t < len_input:
                blur_fft[i,t] = calculate_blur_with_fft(gt_img)
            else:
                pd_mean[i, t], _      = calculate_metrics(pd_img)
                pd_crop_mean[i, t], _ = calculate_metrics(pd_crop_img)
                gt_crop_mean[i, t], _ = calculate_metrics(gt_crop_img)
                dr_mean[i, t], _      = calculate_metrics(dr_img)

                error_gt_pd[i, t]      = calculate_error(gt_mean[i, t], pd_mean[i, t])
                error_crop_gt_dr[i, t] = calculate_error(gt_crop_mean[i, t], dr_mean[i, t])
                error_crop_gt_pd[i, t] = calculate_error(gt_crop_mean[i, t], pd_crop_mean[i, t])

                ssim_gt_pd[i,t]        = calculate_ssim(gt_img, pd_img)
                ssim_crop_gt_dr[i,t]   = calculate_ssim(gt_crop_img, dr_img)
                ssim_crop_gt_pd[i,t]   = calculate_ssim(gt_crop_img, pd_crop_img)
                
                blur_fft[i,t] = calculate_blur_with_fft(pd_img)
                
        # --------------------------------------------------------------------#
        # プロット
        if not args.dont_plot:
            save_dir = os.path.join(dst_dir, 'fulldisk')
            os.makedirs(save_dir, exist_ok=True)

            writer = imageio.get_writer(f'{save_dir}/{i}.mp4', fps=5)

            for t in tqdm(range(len_seq), leave=False, desc='Plot: '):
                fig = plt.figure(figsize=(24,36), tight_layout=True) #(横,縦)
                gspec = gridspec.GridSpec(17, 3) #(縦,横)
                plt.rcParams["font.size"] = 14

                """
                plot_map(fig, gspec[0:3,0], gt_maps[t])
                plot_map(fig, gspec[0:3,1], pd_maps[t])

                intensity_list = [('GT', gt_mean[i], '#1f77b4'), ('Prediction', pd_mean[i], '#ff7f0e')]
                mean_min = min(np.nanmin(gt_mean[i]), np.nanmin(pd_mean[i])) * 0.95
                mean_max = max(np.nanmax(gt_mean[i]), np.nanmax(pd_mean[i])) * 1.05
                plot_metrics(fig, gspec[3,0:3], intensity_list, 'Mean Intensity', 'Intensity', mean_min, mean_max, 0, len_seq, len_input, is_move=True, t=t)

                error_list = [('GT - Prediction', error_gt_pd[i], '#ff7f0e')]
                plot_metrics(fig, gspec[4,0:3], error_list, 'Mean Intensity Error', '%',-20, 20, 0, len_seq, len_input, is_move=True, t=t, is_error=True)

                ssim_list = [('GT - Prediction', ssim_gt_pd[i], '#ff7f0e')]
                plot_metrics(fig, gspec[5,0:3], ssim_list, 'SSIM', 'SSIM',  0.85, 1, 0, len_seq, len_input, is_move=True, t=t)
                """

                # クロップされたマップのプロット
                plot_map(fig, gspec[0:3,0], gt_maps[t], draw_lng=False)
                plot_map(fig, gspec[0:3,1], pd_maps[t], draw_lng=False)
                plot_map(fig, gspec[0:3,2], dr_maps[t], draw_lng=False)

                # 差分マップのプロット
                if t >= len_input:
                    # ピクセルごとの差分
                    diff_max = max(find_max_in_ndarrays(diff_gt_pds), find_max_in_ndarrays(diff_gt_drs))
                    plot_diff(fig, gspec[3:6,1], gspec[6,1], diff_gt_pds[t], diff_max)
                    plot_diff(fig, gspec[3:6,2], gspec[6,2], diff_gt_drs[t], diff_max)

                    # 経度ごとの差分
                    diff_lng_max = max(find_max_in_ndarrays(diff_lng_gt_pds), find_max_in_ndarrays(diff_lng_gt_drs))
                    plot_diff(fig, gspec[7:10,1], gspec[10,1], diff_lng_gt_pds[t], diff_lng_max)
                    plot_diff(fig, gspec[7:10,2], gspec[10,2], diff_lng_gt_drs[t], diff_lng_max)
                    
                """
                # 輝度強度推移グラフ
                intensity_list = [('GT', gt_crop_mean[i], '#1f77b4'), ('Prediction', pd_crop_mean[i], '#ff7f0e'), ('Sunpy', dr_mean[i], '#2ca02c')]
                mean_min = min(np.nanmin(gt_crop_mean[i]), np.nanmin(pd_crop_mean[i]), np.nanmin(dr_mean[i])) * 0.95
                mean_max = max(np.nanmax(gt_crop_mean[i]), np.nanmax(pd_crop_mean[i]), np.nanmax(dr_mean[i])) * 1.05
                plot_metrics(fig, gspec[11:13,0:3], intensity_list, 'Mean Intensity', 'Intensity', mean_min, mean_max, 0, len_seq, len_input, is_move=True, t=t)
                """

                # 輝度強度の誤差推移グラフ
                error_list = [('GT - Prediction', error_crop_gt_pd[i], '#ff7f0e'), ('GT - Sunpy', error_crop_gt_dr[i], '#2ca02c')]
                plot_metrics(fig, gspec[11:13,0:3], error_list, 'Error of Mean Intensity', '%',-30, 30, 12, len_seq, len_input, is_move=True, t=t, is_error=True)

                # 経度ごとの平均輝度の誤差推移グラフ
                cmap_tab10 = cm.get_cmap('tab10', 10)
                error_list = [('GT - PD | 54 to 90', gt_pd_error_lng[i,:,0], cmap_tab10(0), '-', 'o'),
                              ('GT - PD | 18 to 54', gt_pd_error_lng[i,:,1], cmap_tab10(1), '-', 'o'),
                              ('GT - PD | -18 to 18' , gt_pd_error_lng[i,:,2], cmap_tab10(2), '-', 'o'),
                              ('GT - PD | -18 to -54'  , gt_pd_error_lng[i,:,3], cmap_tab10(3), '-', 'o'),
                              ('GT - PD | -54 to -90'  , gt_pd_error_lng[i,:,4], cmap_tab10(4), '-', 'o'),
                              ('GT - SP | 54 to 90', gt_dr_error_lng[i,:,0], cmap_tab10(0), ':', 'x'),
                              ('GT - SP | 18 to 54', gt_dr_error_lng[i,:,1], cmap_tab10(1), ':', 'x'),
                              ('GT - SP | -18 to 18' , gt_dr_error_lng[i,:,2], cmap_tab10(2), ':', 'x'),
                              ('GT - SP | -18 to -54'  , gt_dr_error_lng[i,:,3], cmap_tab10(3), ':', 'x'),
                              ('GT - SP | -54 to -90'  , gt_dr_error_lng[i,:,4], cmap_tab10(4), ':', 'x')]
                plot_metrics(fig, gspec[13:15,0:3], error_list, 'Error rate per longitude ', '%', -60, 60, 12, len_seq, len_input, is_move=True, t=t, is_error=True)
                
                # ブラー推移グラフ
                plot_metrics(fig, gspec[15:17,0:3], [('Blur', blur_fft[i], '#1f77b4')], 'Blur', 'Blur', 0, 10, 0, len_seq, len_input, is_move=True, t=t)

                """
                # SSIM推移グラフ
                ssim_list = [('GT - Prediction', ssim_crop_gt_pd[i], '#ff7f0e'), ('GT - Sunpy', ssim_crop_gt_dr[i], '#2ca02c')]
                plot_metrics(fig, gspec[5,3:6], ssim_list, 'SSIM', 'SSIM',  0.85, 1, 0, len_seq, len_input, is_move=True, t=t)
                """

                writer.append_data(fig_to_ndarray(fig))
                plt.close(fig)
                plt.clf()

            writer.close()


    # ------------------------------------------------------------------------------------------------------------------------
    # まとめプロット
    # ------------------------------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(20,48), tight_layout=True)
    gspec = gridspec.GridSpec(5, 2)
    plt.rcParams["font.size"] = 12

    mean_error_gt_pd = np.nanmean(np.abs(error_crop_gt_pd), axis=0)
    mean_error_gt_dr = np.nanmean(np.abs(error_crop_gt_dr), axis=0)
    mean_ssim_gt_pd  = np.nanmean(ssim_crop_gt_pd, axis=0)
    mean_ssim_gt_dr  = np.nanmean(ssim_crop_gt_dr, axis=0)
    mean_error_gt_pd_lng = np.nanmean(gt_pd_error_lng, axis=0)
    mean_error_gt_dr_lng = np.nanmean(gt_dr_error_lng, axis=0)

    """
    # シンプルなGT-PDの平均輝度の誤差推移グラフ
    error_list = [('GT - Prediction', mean_error_gt_pd, '#ff7f0e')]
    plot_metrics(fig, gspec[0,:], error_list, 'Average Absolute Error of Mean Intensity', '%',0, 10, len_input, len_seq, len_input, is_write_val=True)

    # シンプルなGT-SPの平均輝度の誤差推移グラフ
    error_list = [('GT - Sunpy', mean_error_gt_dr, '#2ca02c')]
    plot_metrics(fig, gspec[1,:], error_list, 'Average Absolute Error of Mean Intensity', '%',0, 20, len_input, len_seq, len_input, is_write_val=True)
    """

    # GTと動画予測の平均輝度の誤差推移グラフ
    title = 'Average Error Rate of Mean Intensity'
    error_list = [('GT - Prediction', mean_error_gt_pd, '#ff7f0e'), ('GT - Sunpy', mean_error_gt_dr, '#2ca02c')]
    plot_metrics(fig, gspec[0,:], error_list, title, '%',0, 20, len_input, len_seq, len_input, is_write_val=True)

    # GTと動画予測の平均輝度の誤差推移グラフ
    cmap_tab10 = cm.get_cmap('tab10', 10)
    title = 'Average Error Rate of Mean Intensity Per Longitude of GT - Prediction'
    error_list = [('54d to 90d',   mean_error_gt_pd_lng[:,0], cmap_tab10(0), '-', 'o'),
                  ('18d to 54d',   mean_error_gt_pd_lng[:,1], cmap_tab10(1), '-', 'o'),
                  ('-18d to 18d',  mean_error_gt_pd_lng[:,2], cmap_tab10(2), '-', 'o'),
                  ('-54d to -18d', mean_error_gt_pd_lng[:,3], cmap_tab10(3), '-', 'o'),
                  ('-90d to -54d', mean_error_gt_pd_lng[:,4], cmap_tab10(4), '-', 'o')]
    plot_metrics(fig, gspec[1,:], error_list, title, '%', -15, 15, len_input, len_seq, len_input, is_write_val=True)

    # GTと差動回転モデルの平均輝度の誤差推移グラフ
    title = 'Average Error Rate of Mean Intensity Per Longitude of GT - Diffrential Rotation Model by Sunpy'
    error_list = [('54d to 90d',   mean_error_gt_dr_lng[:,0], cmap_tab10(0), ':', 'x'),
                  ('18d to 54d',   mean_error_gt_dr_lng[:,1], cmap_tab10(1), ':', 'x'),
                  ('-18d to 18d',  mean_error_gt_dr_lng[:,2], cmap_tab10(2), ':', 'x'),
                  ('-54d to -18d', mean_error_gt_dr_lng[:,3], cmap_tab10(3), ':', 'x'),
                  ('-90d to -54d', mean_error_gt_dr_lng[:,4], cmap_tab10(4), ':', 'x')]
    plot_metrics(fig, gspec[2,:], error_list, title, '%', -70, 70, len_input, len_seq, len_input, is_write_val=True)

    # SSIM
    ssim_list = [('GT - Prediction', mean_ssim_gt_pd, '#ff7f0e'), ('GT - Sunpy', mean_ssim_gt_dr, '#2ca02c')]
    plot_metrics(fig, gspec[3,:], ssim_list, 'Average SSIM', 'SSIM',  0.85, 1, len_input, len_seq, len_input, is_write_val=True)

    # 東のリムの平均輝度の再現度の評価
    min_v = min(np.nanmin(gt_east_limb_last_input), np.nanmin(gt_east_limb_last_output), np.nanmin(pd_east_limb_last_output)) * 0.8
    max_v = max(np.nanmax(gt_east_limb_last_input), np.nanmin(gt_east_limb_last_output), np.nanmax(pd_east_limb_last_output)) * 1.1

    #   GTの最終出力とPDの最終出力の散布図
    title = 'Mean Intensity of East Limb'
    plot_scatter(fig, gspec[4,0], gt_east_limb_last_output, pd_east_limb_last_output, min_v, max_v, 'GT Intensity at the Last', 'Prediction Intensity at the Last')
    
    #   GTの最終入力とGTの最終出力の散布図
    plot_scatter(fig, gspec[4,1], gt_east_limb_last_output, gt_east_limb_last_input, min_v, max_v, 'GT Intensity at the Last', 'INPUT IMAGE Intensity at the Last')

    summary_plot_path = os.path.join(dst_dir, 'summary_fulldisk.png')
    fig.savefig(summary_plot_path)