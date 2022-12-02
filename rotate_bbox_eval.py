from glob import glob
import random
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.coordinates import RotatedSunFrame
import sunpy.coordinates.frames as f

fits_dirs = sorted(glob('fits_original/*'))
npy_files  = sorted(glob('npy_decoded/*'))

rectangle_coords = [
#           [ -650, 500, -900, 250 ],
#            [ 500, -100, -500, -600 ],
            [ 200, -150, -500, -620 ],
#            [ -700, -250, -850, -400 ],
        ]


for i in range(len(npy_files)):

    # load images
    fits_files = sorted(glob(fits_dirs[i] + '/*'))
    video_prediction = np.load(npy_files[i])
    start_date = sunpy.map.Map(fits_files[0]).date
    
    # set durations
    durations = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44,] * u.hour
    obstime = [start_date] * len(durations)
    start_top_right_x   = [rectangle_coords[i][0]] * len(durations) * u.arcsec
    start_top_right_y   = [rectangle_coords[i][1]] * len(durations) * u.arcsec
    start_bottom_left_x = [rectangle_coords[i][2]] * len(durations) * u.arcsec
    start_bottom_left_y = [rectangle_coords[i][3]] * len(durations) * u.arcsec

    # 観測者の位置も変化する。異なる観測者時間の座標フレームを定義し、それぞれで時間経過させ、差動回転を表示する。
    start_array_top_right   = f.Helioprojective(start_top_right_x,   start_top_right_y,   obstime=obstime, observer="earth")
    start_array_bottom_left = f.Helioprojective(start_bottom_left_x, start_bottom_left_y, obstime=obstime, observer="earth")

    ar_array_top_right    = RotatedSunFrame(base=start_array_top_right, duration=durations)
    ar_array_bottom_left  = RotatedSunFrame(base=start_array_bottom_left, duration=durations)

    earth_hpc_top_right   = f.Helioprojective(obstime=ar_array_top_right.rotated_time, observer="earth")
    earth_hpc_bottom_left = f.Helioprojective(obstime=ar_array_bottom_left.rotated_time, observer="earth")

    coords_top_right = SkyCoord(ar_array_top_right.transform_to(earth_hpc_top_right))
    coords_bottom_left = SkyCoord(ar_array_bottom_left.transform_to(earth_hpc_bottom_left))

    fig = plt.figure(figsize=(30, 60), dpi=300)
    gs = fig.add_gridspec(49, 29)
    num_ax = 8 
    ax = []
    ax.append(fig.add_subplot(gs[0, 1:9], xlim=(0,10),ylim=(0,10)))
    ax.append(fig.add_subplot(gs[0, 13:21],xlim=(0,10),ylim=(0,10)))
    ax[0].xaxis.set_major_locator(mpl.ticker.NullLocator())
    ax[0].yaxis.set_major_locator(mpl.ticker.NullLocator())
    ax[1].xaxis.set_major_locator(mpl.ticker.NullLocator())
    ax[1].yaxis.set_major_locator(mpl.ticker.NullLocator())

    ax[0].text(1,15, 'GT', fontsize=18)
    ax[1].text(1,15, 'Prediction', fontsize=18)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    # plots
    for j in range(len(durations)):
        fits_data = fits.open(fits_files[j])
        fulldisk_original = np.array(fits_data[1].data)
        header = fits_data[1].header
        fulldisk_prediction = video_prediction[j]

        # set coordinates
        map_original  = sunpy.map.Map(fits_files[j])
        map_prediction = sunpy.map.Map(fulldisk_prediction, header)

        original_pixel_top_right = map_original.world_to_pixel(coords_top_right[j])
        original_pixel_bottom_left = map_original.world_to_pixel(coords_bottom_left[j])

        prediction_pixel_top_right = map_prediction.world_to_pixel(coords_top_right[j])
        prediction_pixel_bottom_left = map_prediction.world_to_pixel(coords_bottom_left[j])

        # cut off regions
        region_original   = np.ravel(fulldisk_original[round(original_pixel_bottom_left.x.value):round(original_pixel_top_right.x.value),
                                        round(original_pixel_bottom_left.y.value):round(original_pixel_top_right.y.value)])
        region_prediction = np.ravel(fulldisk_prediction[round(prediction_pixel_bottom_left.x.value):round(prediction_pixel_top_right.x.value),
                                           round(prediction_pixel_bottom_left.y.value):round(prediction_pixel_top_right.y.value)])
 
        # calculate indices
        original_region_max   = np.amax(region_original)
        original_region_avg   = np.mean(region_original)
        prediction_region_max = np.amax(region_prediction)
        prediction_region_avg = np.mean(region_prediction)

        original_fulldisk_max   = np.amax(fulldisk_original)
        original_fulldisk_avg   = np.mean(fulldisk_original)
        prediction_fulldisk_max = np.amax(fulldisk_prediction)
        prediction_fulldisk_avg = np.mean(fulldisk_prediction)

        print('t =',j)
        print(map_original.date)
        print(original_pixel_bottom_left)
        print(original_pixel_top_right)
        print(region_original.size)
        print(region_prediction.size)
        print(fulldisk_original.size)
        print(fulldisk_prediction.size)
 
        print('max', original_region_max)
        print('avg', original_region_avg)
        print('max', prediction_region_max)
        print('avg', prediction_region_avg)
        print('------')

        # index
        ax.append(fig.add_subplot(gs[2+j*4, 0]))
        ax[2 + j*num_ax].text(0,0.5,'t = ' + str(j+13), fontsize=20)
        ax[2 + j*num_ax].axis("off")
        ax[2 + j*num_ax].spines['right'].set_visible(False)
        ax[2 + j*num_ax].spines['left'].set_visible(False)
        ax[2 + j*num_ax].spines['bottom'].set_visible(False)
        ax[2 + j*num_ax].spines['top'].set_visible(False)


        # plot map
        ax.append(fig.add_subplot(gs[1+j*4:5+j*4, 1:5], projection=map_original))
        ax.append(fig.add_subplot(gs[1+j*4:5+j*4, 13:17], projection=map_prediction))
        map_original.plot(axes=ax[3 + j*num_ax], clip_interval=(1, 99.99)*u.percent)
        map_prediction.plot(axes=ax[4 + j*num_ax], clip_interval=(1, 99.99)*u.percent)
        map_original.draw_quadrangle(
            coords_bottom_left[j],
            axes=ax[3 + j*num_ax],
            top_right=coords_top_right[j],
            edgecolor="blue",
            linestyle="--",
            linewidth=2,
        )
        map_prediction.draw_quadrangle(
            coords_bottom_left[j],
            axes=ax[4+j*num_ax],
            top_right=coords_top_right[j],
            edgecolor="blue",
            linestyle="--",
            linewidth=2,
        )
        ax[3 + j*num_ax].axis("off")
        ax[4 + j*num_ax].axis("off")

        # hist of region
        max_x_region = max(original_region_max, prediction_region_max)
        class_width = max_x_region / 50
        bins_original_region   = round(original_region_max / class_width)
        bins_prediction_region = round(prediction_region_max / class_width)
        freq_original_region   = np.histogram(region_original, bins=bins_original_region, density=False)
        freq_prediction_region = np.histogram(region_prediction, bins=bins_prediction_region, density=False)
        max_y_region = max(np.amax(freq_original_region[0]), np.amax(freq_prediction_region[0])) * 1.10

        ax.append(fig.add_subplot(gs[1+j*4:5+j*4, 5:9], xlim=(0,max_x_region), ylim=(1,max_y_region)))
        ax[5 + j*num_ax].hist(region_original, bins=bins_original_region)
        ax[5 + j*num_ax].text(max_x_region * 0.4, max_y_region * 0.8, 'max: ' + str(original_region_max)[:6])
        ax[5 + j*num_ax].text(max_x_region * 0.4, max_y_region * 0.7, 'avg: ' + str(original_region_avg)[:6])
        ax[5 + j*num_ax].set_title('brightness distribution (region)')
        ax[5 + j*num_ax].set_box_aspect(1)

        ax.append(fig.add_subplot(gs[1+j*4:5+j*4, 17:21], xlim=(0,max_x_region), ylim=(1,max_y_region)))
        ax[6 + j*num_ax].hist(region_prediction, bins=bins_prediction_region)
        ax[6 + j*num_ax].text(max_x_region * 0.4, max_y_region * 0.8, 'max: ' + str(prediction_region_max)[:6])
        ax[6 + j*num_ax].text(max_x_region * 0.4, max_y_region * 0.7, 'avg: ' + str(prediction_region_avg)[:6])
        ax[6 + j*num_ax].set_title('brightness distribution (region)')
        ax[6 + j*num_ax].set_box_aspect(1)

        # hist of fulldisk
        max_x_fulldisk = max(original_fulldisk_max, prediction_fulldisk_max)
        class_width = max_x_fulldisk / 50
        bins_original_fulldisk   = round(original_fulldisk_max / class_width)
        bins_prediction_fulldisk = round(prediction_fulldisk_max / class_width)
        freq_original_fulldisk   = np.histogram(fulldisk_original, bins=bins_original_fulldisk, density=False)
        freq_prediction_fulldisk = np.histogram(fulldisk_prediction, bins=bins_prediction_fulldisk, density=False)
        max_y_fulldisk = max(np.amax(freq_original_fulldisk[0]), np.amax(freq_prediction_fulldisk[0])) * 1.10

        ax.append(fig.add_subplot(gs[1+j*4:5+j*4, 9:13], xlim=(0, max_x_fulldisk), ylim=(1, max_y_fulldisk)))
        ax[7 + j*num_ax].hist(fulldisk_original, bins=bins_original_fulldisk)
        ax[7 + j*num_ax].text(max_x_fulldisk * 0.4, max_y_fulldisk * 0.8, 'max: ' + str(original_fulldisk_max)[:6])
        ax[7 + j*num_ax].text(max_x_fulldisk * 0.4, max_y_fulldisk * 0.7, 'avg: ' + str(original_fulldisk_avg)[:6])
        ax[7 + j*num_ax].set_title('brightness distribution (fulldisk)')
        ax[7 + j*num_ax].set_xlabel('brightness')
        ax[7 + j*num_ax].set_box_aspect(1)

        ax.append(fig.add_subplot(gs[1+j*4:5+j*4, 21:25], xlim=(0, max_x_fulldisk), ylim=(1, max_y_fulldisk)))
        ax[8 + j*num_ax].hist(fulldisk_prediction, bins=bins_prediction_fulldisk)
        ax[8 + j*num_ax].text(max_x_fulldisk * 0.4, max_y_fulldisk * 0.8, 'max: ' + str(prediction_fulldisk_max)[:6])
        ax[8 + j*num_ax].text(max_x_fulldisk * 0.4, max_y_fulldisk * 0.7, 'avg: ' + str(prediction_fulldisk_avg)[:6])
        ax[8 + j*num_ax].set_title('brightness distribution (fulldisk)')
        ax[8 + j*num_ax].set_xlabel('brightness')
        ax[8 + j*num_ax].set_box_aspect(1)

        """
        ax.append(fig.add_subplot(gs[1+j*4:5+j*4, 9:13], xlim=(0, 1), ylim=(0, 1)))
        ax.append(fig.add_subplot(gs[1+j*4:5+j*4, 21:25], xlim=(0, 1), ylim=(0, 1)))
        ax[7 + j*num_ax].set_box_aspect(1)
        ax[8 + j*num_ax].set_box_aspect(1)
        """

        # scatter
        scatter_max = max(original_region_max, prediction_region_max)
        r = np.corrcoef(region_original, region_prediction)[0,1]

        ax.append(fig.add_subplot(gs[1+j*4:5+j*4, 25:29], xlim=(0, scatter_max), ylim=(0, scatter_max)))
        ax[9 + j*num_ax].scatter(region_original, region_prediction, s=10, edgecolors='red' ,c='pink', alpha=0.3)
        ax[9 + j*num_ax].grid()
        ax[9 + j*num_ax].set_aspect('equal')
        ax[9 + j*num_ax].text(scatter_max * 0.2, scatter_max* 0.8, 'r = ' + str(r)[:6])
        ax[9 + j*num_ax].set_title('brightness of the same pixel')
        ax[9 + j*num_ax].set_xlabel('GT')
        ax[9 + j*num_ax].set_ylabel('Prediction')

    fig.tight_layout()
    fig.savefig('map_image/comparison_' + str(i) + '.png')
