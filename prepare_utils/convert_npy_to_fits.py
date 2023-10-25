import os
import glob
import shutil
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import sunpy.map

def convert_npy_to_fits(timestamp):
    print('5. convert npy to fits....')
    # Get list of directories in tmp
    child_dirs = sorted([d for d in os.listdir(f'{timestamp}/tmp') if os.path.isdir(os.path.join(timestamp, 'tmp', d))])

    for child_dir in tqdm(child_dirs):
        # Get list of npy files in each directory
        npy_files = sorted(glob.glob(f'{timestamp}/tmp/{child_dir}/*.npy'))

        # 対応するgt_fitsディレクトリにあるfitsファイルのリストを取得しソート
        fits_files = sorted(glob.glob(f'{timestamp}/gt_fits/{child_dir}/*.fits'))

        for i, npy_file in enumerate(npy_files):
            # Load npy file

            # 対応するfitsファイルのヘッダーを読み込む
            # 12はinput_lengthに対応!!!!
            fits_file = fits_files[12 + i]

            """
            gt_map = sunpy.map.Map(fits_file)
            data = np.load(npy_file)
            print(data.shape)

            new_map = sunpy.map.Map(data, gt_map.meta)

            pd_dir = f'{timestamp}/pd_fits/{child_dir}'
            os.makedirs(pd_dir, exist_ok=True)

            new_map.save(f'{pd_dir}/{os.path.basename(fits_file)}', overwrite=True)



            """
            hdu_list = fits.open(fits_file)

            data = np.load(npy_file)
            header = hdu_list[1].header.copy()

            hdu_list[1] = fits.CompImageHDU(data=data, header=header)

            # Save as new fits file
            pd_dir = f'{timestamp}/pd_fits/{child_dir}'
            os.makedirs(pd_dir, exist_ok=True)


            # GTのFITSファイルと同じ名前で保存
            hdu_list.writeto(f'{pd_dir}/{os.path.basename(fits_file)}',overwrite=True)


    shutil.rmtree(f'{timestamp}/tmp')
