import os
import glob
import shutil
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from sunpy.map import Map

def convert_npy_to_fits(tmp_dir, gt_fits_dir, pd_fits_dir):
    print('4. convert npy to fits....')
    # Get list of directories in tmp
    child_dirs = sorted([d for d in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, d))])

    for child_dir in tqdm(child_dirs):
        # Get list of npy files in each directory
        npy_files = sorted(glob.glob(f'{tmp_dir}/{child_dir}/*.npy'))

        # 対応するgt_fitsディレクトリにあるfitsファイルのリストを取得しソート
        fits_files = sorted(glob.glob(f'{gt_fits_dir}/{child_dir}/*.fits'))

        for i, npy_file in enumerate(npy_files):

            data = np.load(npy_file)
            data = np.squeeze(data)

            # 対応するfitsファイルのヘッダーを読み込む
            # 12はinput_lengthに対応!!!!
            fits_file = fits_files[12 + i]
            meta = Map(fits_file).meta

            new_map = Map(data, meta)

            # Save as new fits file
            pd_dir = f'{pd_fits_dir}/{child_dir}'
            os.makedirs(pd_dir, exist_ok=True)

            # GTのFITSファイルと同じ名前で保存
            new_map_path = f'{pd_dir}/{os.path.basename(fits_file)}'
            new_map.save(new_map_path, filetype='fits', overwrite=True)


    shutil.rmtree(f'{tmp_dir}')