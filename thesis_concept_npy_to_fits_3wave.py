import os
import glob
import numpy as np
from sunpy.map import Map


def get_exptime(fits_file):
    exptime = Map(fits_file).meta['EXPTIME']
    return exptime
    

def decode_preprocessing(data, exptime, min_value, max_value):
    # Decode the scaling
    data = np.square(data)

    # Decode the normalization
    data = data * (max_value - min_value) + min_value

    # Decode the exptime normalization
    dcata = data / 3.000 * exptime

    return data


npy_file = './concept_exp2/tmp/01.npy'

data = np.load(npy_file)
data = np.squeeze(data)

print(data.shape)

for w, wave in enumerate(['211', '193', '171']):

    fits_files = sorted(glob.glob(f'./concept_exp2/{wave}/gt/*.fits'))
    data_wave = data[:, :, :, w]

    for i in range(12):
        gt_fits_file = fits_files[i+12]
        
        map_image = data_wave[i]
        map_image = decode_preprocessing(map_image, get_exptime(gt_fits_file), 0, 10000)

        map_meta = Map(gt_fits_file).meta
        
        pd_map = Map(map_image, map_meta)
        # 保存
        pd_map.save(f'./concept_exp2/{wave}/pd/{os.path.basename(gt_fits_file)}', filetype='fits', overwrite=True)
        
        