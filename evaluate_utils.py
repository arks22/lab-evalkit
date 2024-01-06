import argparse
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import warnings
import imageio
from skimage.metrics import structural_similarity

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib

from astropy.io import fits
import astropy.units as u
from astropy.time import TimeDelta
from astropy.coordinates import SkyCoord

from matplotlib import rc
rc('font', family='serif')
rc('font', serif='Times New Roman')
rc('text', usetex=False)

from sunpy.map import Map
from sunpy.coordinates import frames
from sunpy.physics.differential_rotation import differential_rotate
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates import sun
from astropy.coordinates import Angle

# cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

def rotate_map(smap, hour):
    time_diff = TimeDelta(hour*60*60, format='sec')  # 4時間を秒に換算
    # 差動回転の適用
    earth = get_body_heliographic_stonyhurst('earth', time=smap.date + time_diff)
    rotated_map = differential_rotate(smap, observer=earth)

    return rotated_map

def resize_map(smap, new_shape):
    """
    Resizes a SunPy Map to a new shape.

    Parameters:
    mymap (sunpy.map.Map): The original SunPy Map.
    new_shape (tuple): The new shape for the map (height, width).

    Returns:
    sunpy.map.Map: The resampled map with the new shape.
    """

    # astropy Quantityオブジェクトに新しいシェイプを変換
    new_dimensions = u.Quantity(new_shape, u.pixel)

    # データをリサイズ
    resampled_map = smap.resample(new_dimensions)

    return resampled_map

def mask_map(map1, map2):
    map1_data = map1.data.copy().astype(np.float32)
    map2_data = map2.data.copy().astype(np.float32)
    map1_data[np.isnan(map2_data)] = np.nan

    masked_map = Map(map1_data, map1.meta)

    return masked_map

def generate_submap_by_lng(smap, lon_tuple):
    """
    与えられた経度範囲に基づいてsubmapを作成

    引数:
        smap: sunpy.map.Mapオブジェクト
        lon_start: 開始経度(int)
        lon_end: 終了経度(int)

    返り値:
        選択した経度範囲のsunpy.map.Map
    """
    # 入力マップから全てのピクセルの経度値を取得
    all_coords = smap.pixel_to_world(np.arange(smap.data.shape[1])[np.newaxis, :] * u.pix,
                                     np.arange(smap.data.shape[0])[:, np.newaxis] * u.pix)
    #座標系を HeliographicStonyhurst に変換
    all_lons = all_coords.transform_to(frames.HeliographicStonyhurst).lon

    # マスクを作成: 始点と終点の経度の間にあるピクセルを選択
    lon_start = Angle(lon_tuple[0] * u.deg)
    lon_end = Angle(lon_tuple[1] * u.deg)
    mask = np.logical_and(all_lons >= lon_start, all_lons <= lon_end)

    # 新しいデータ配列を作成
    new_data = np.full_like(smap.data, np.nan)
    new_data[mask] = smap.data[mask]

    # マスクを適用した新しいマップを作成
    new_map = Map(new_data, smap.meta)

    return new_map


def generate_mean_map_by_lng(smap, num_regions, area_debug=False):
    """
    与えられた太陽マップを経度に基づいて分割し、各領域の平均輝度をその領域のピクセルに代入した新しいマップを生成する。

    引数:
    smap: sunpy Map object
        調査する太陽マップ。
    n_regions: int
        太陽マップを分割する領域の数。

    返り値:
    sunpy Map object
        平均輝度が領域ごとに代入された新しいマップ。
    ndarray
        各領域の平均値が格納されたnumpy配列
    """

    mean_region_data = np.full_like(smap.data, np.nan)

    deg_east, deg_west = -90*u.deg, 90*u.deg
    lon_step = (deg_west - deg_east) / num_regions 

    mean_arr = np.zeros((num_regions))

    # 入力マップから全てのピクセルの経度値を取得
    all_coords = smap.pixel_to_world(np.arange(smap.data.shape[1])[np.newaxis, :] * u.pix,
                                     np.arange(smap.data.shape[0])[:, np.newaxis] * u.pix)
    #座標系を HeliographicStonyhurst に変換
    all_lons = all_coords.transform_to(frames.HeliographicStonyhurst).lon

    for i in range(num_regions):
        lon_start = deg_east + i * lon_step
        lon_end = lon_start + lon_step

        # マスクを作成: 始点と終点の経度の間にあるピクセルを選択
        mask = np.logical_and(all_lons >= lon_start, all_lons <= lon_end)
        # マスクした領域にあるsmapの平均を算出
        mean_region_mean = np.nanmean(smap.data[mask])
        mean_arr[i] = mean_region_mean
        # mean_region_dataのマスクした領域に平均値を代入
        mean_region_data = np.where(mask, mean_region_mean, mean_region_data)

        if area_debug:
            not_nan_mask = ~np.isnan(smap.data)
            combined_mask = np.logical_and(not_nan_mask, mask)
            print('---')
            print(np.count_nonzero(combined_mask))
            print(np.count_nonzero(mask))
            print('------------------')

    mean_region_map = Map(mean_region_data, smap.meta)

    return mean_region_map, mean_arr


def generate_diff_map(map1, map2):
    diff_data = map1.data - map2.data
    diff_data = np.sign(diff_data) * np.log1p(np.abs(diff_data))

    diff_map = Map(diff_data, map1.meta)

    return diff_map


def find_limb_deg(smap):
    """
    太陽の東と西のリムの角度を特定する
    引数:
       - sun_map (sunpy.map.Map): 太陽の画像データ
    返り値:
      - 左端および右端のx座標のタプル(u.deg)
    """

    # 入力マップから全てのピクセルの経度値を取得
    all_coords = smap.pixel_to_world(np.arange(smap.data.shape[1])[np.newaxis, :] * u.pix,
                                     np.arange(smap.data.shape[0])[:, np.newaxis] * u.pix)
    #座標系を HeliographicStonyhurst に変換
    all_coords = all_coords.transform_to(frames.HeliographicStonyhurst)
    all_lons = all_coords.lon

    min_lon = np.nanmin(all_lons)
    max_lon = np.nanmax(all_lons)

    return min_lon, max_lon



def calculate_metrics(img):
    # 画像の輝度の平均値と最大値を計算
    mean_val = np.nanmean(img)
    max_val = np.nanmax(img)

    return mean_val, max_val


def calculate_error(x, y):
    return 100 * (x - y) / x 


def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def calculate_ssim(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    #NaNは0埋めで対応
    img1 = np.where(np.isnan(img1), 0, img1)
    img2 = np.where(np.isnan(img2), 0, img2)

    data_range = max(np.nanmax(img1), np.nanmax(img2))

    ssim_val = structural_similarity(img1, img2, data_range=data_range)

    return ssim_val
    

def calculate_blur_with_fft(img):
    # フーリエ変換を行う
    f = np.fft.fft2(img)
    # 画像の中心に低周波数の成分がくるように並べかえる
    fshift = np.fft.fftshift(f)

    # マグニチュードスペクトルを計算
    magnitude_spectrum = np.log(np.abs(fshift))
 
    # 低周波成分を除去
    filter_size = 200
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    magnitude_spectrum[crow-filter_size:crow+filter_size, ccol-filter_size:ccol+filter_size] = 0

    # ぼやけ度を計算（高周波成分のエネルギーの平均）
    blur_measure = np.mean(magnitude_spectrum)

    return blur_measure
    

def find_max_in_ndarrays(A):
    max_values = [np.nanmax(np.abs(arr)) for arr in A]
    max_value = max(max_values)

    return max_value


def fig_to_ndarray(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_fits_img(fits_file):
    img = np.array(fits.open(fits_file)[1].data)
    return img


def plot_images(maps, tilte):
    fig = plt.figure(figsize=(len(maps)*8, 8))
    plt.subplots_adjust(wspace=0.1)
    
    for i, m in enumerate(maps):
        ax = fig.add_subplot(1, len(maps), i+1, projection=m.wcs)  # 個別にprojectionを設定
        m.plot(axes=ax, annotate=False)  # MapオブジェクトごとにWCSを設定
        ax.axis('off')
        ax.set_title(f't = {i}\n( {(i)*4}h )', fontsize=36)  # サブプロットのタイトルに番号を設定

    plt.subplots_adjust(top=0.8)
    plt.tight_layout(pad=2.0)

    plt.savefig(tilte)


def plot_map(fig, fig_gspec, smap, draw_lng=False):
    ax = fig.add_subplot(fig_gspec, projection=smap)
    smap.plot(clip_interval=(1,99.99)*u.percent)

    if draw_lng:
        longitude_lines = np.arange(-90, -53, 36)  # -90度から90度まで、36度ごと
        for lon in longitude_lines:
            coord = SkyCoord(lon*u.deg, np.linspace(-90, 90, 100)*u.deg, frame=frames.HeliographicStonyhurst, observer=smap.observer_coordinate)
            ax.plot_coord(coord, color='blue', linewidth=3)

    """
    if draw_bbox:
        smap.draw_quadrangle(bottom_left=bottom_left, axes=ax, top_right=top_right, linestyle="--", linewidth=2)
    """


def plot_diff(fig, diff_gspec, cbar_gspec, diff, vmax):
    ax = fig.add_subplot(diff_gspec)
    cmap = plt.get_cmap('RdBu_r')
    im = ax.imshow(diff, cmap=cmap, interpolation='none', origin='lower', vmin=-vmax, vmax=vmax)
    ax.set_xticks([])  # Disable x-axis ticks
    ax.set_yticks([])  # Disable y-axis ticks

    # カラーバーの設定
    cax = fig.add_subplot(cbar_gspec) # Add a new subplot for colorbar
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal') # Set orientation to horizontal

    ticks = np.linspace(-vmax, vmax, 7)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(['{:0.0e}'.format(val) for val in np.sign(ticks) * np.expm1(np.abs(ticks))])


def plot_histogram(fig, fig_gspec, img, min_v, max_v):
    img = np.ravel(img)
    ax = fig.add_subplot(fig_gspec)  # Share y-axis with ax1
    ax.hist(img, bins=20, range=(min_v, max_v), color='blue', alpha=0.7, log=True)
    ax.set_ylim(1, 10**8)
    ax.set_title('Intensity Histogram')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Frequency')
    ax.grid(True)


def plot_scatter(fig, fig_gspec, x, y, x_label, y_label, lim=None):
    ax = fig.add_subplot(fig_gspec) 

    correlation_coefficient = np.corrcoef(x, y)[0, 1]
    ax.scatter(x,y, marker='x', color='green', label=f'Correlation: {correlation_coefficient:.2f}')

    ax.legend()
    ax.grid(True)
    if lim is not None:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[0], lim[1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def plot_scatter_thesis(title, x, y, x_label, y_label, color='green', lim=None):
    matplotlib.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)

    correlation_coefficient = np.corrcoef(x, y)[0, 1]
    ax.scatter(x,y, marker='x', color=color, label=f'Correlation: {correlation_coefficient:.2f}')

    ax.legend()
    ax.grid(True)
    if lim is not None:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[0], lim[1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    plt.tight_layout()
    plt.savefig(title + '.png')


def plot_metrics(fig, fig_gspec, metrics_list, title, ylabel, start_seq, len_seq, len_input, y_lim=None, is_move=False, t=0, is_error=False, is_write_val=False):
    ax = fig.add_subplot(fig_gspec)
    for j, metrics in enumerate(metrics_list):

        if len(metrics) > 3:
            linestyle = metrics[3]
            marker = metrics[4]
        else:
            linestyle = '-'
            marker = 'o'

        ax.plot(metrics[1], color=metrics[2], label=metrics[0], linewidth=2, linestyle=linestyle, marker=marker)

        if is_write_val: # データ点にテキストを追加
            x_val = np.arange(len(metrics[1]))
            y_val = metrics[1]
            if y_lim is None:
                raise ValueError('y_lim must be set when is_write_val is True')
            text_margin = (y_lim[1]- y_lim[0]) / 20
            for i in range(len(x_val)):
                if i > start_seq:
                    plt.text(x_val[i], y_val[i] + text_margin, f'{y_val[i]:.3f}', ha='center', va='center')

    ax.set_xlim(start_seq, len_seq)
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_xlabel('t')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_metrics_thesis(metrics_list, title, ylabel, start_seq, len_seq, len_input, y_lim=None, is_move=False, t=0, is_error=False, is_write_val=False, square=False):
    matplotlib.rcParams.update({'font.size': 20})
    if square:
        fig = plt.figure(figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(1, 1, 1)
    for j, metrics in enumerate(metrics_list):

        if len(metrics) > 3:
            linestyle = metrics[3]
            marker = metrics[4]
        else:
            linestyle = '-'
            marker = 'o'

        ax.plot(metrics[1], color=metrics[2], label=metrics[0], linewidth=2, linestyle=linestyle, marker=marker)

        if is_write_val: # データ点にテキストを追加
            x_val = np.arange(len(metrics[1]))
            y_val = metrics[1]
            if y_lim is None:
                raise ValueError('y_lim must be set when is_write_val is True')
            text_margin = (y_lim[1]- y_lim[0]) / 20
            for i in range(len(x_val)):
                if i > start_seq:
                    plt.text(x_val[i], y_val[i] + text_margin, f'{y_val[i]:.3f}', ha='center', va='center')
    if is_error:
        ax.axhline(y=0, color='blue')
        #ax.axhspan(-5, 5, color="blue", alpha=0.1)

    ax.axvspan(0, len_input - 0.5, color="gray", alpha=0.1)
    ax.grid(True)
    ax.legend()

    ax.set_xlim(start_seq, len_seq)
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_xlabel('t')
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(title + '.png')