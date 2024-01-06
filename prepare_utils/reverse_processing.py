import numpy as np
import glob
import os
from tqdm import tqdm
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


# FITSのヘッダーの露光時間でデコードする必要があるため最後
def reverse_processing(pd_fits_dir, min_value, max_value):
    print('5. reverse processing....')
    pd_child_dirs = sorted(os.listdir(pd_fits_dir))
    print(pd_child_dirs)

    for dir in tqdm(pd_child_dirs):
        pd_fits_files = sorted(glob.glob(f'{pd_fits_dir}/{dir}/*.fits'))

        for pd_fits_file in pd_fits_files:
            
            pd_data = Map(pd_fits_file).data
            pd_meta = Map(pd_fits_file).meta
            
            exptime = get_exptime(pd_fits_file)

            # Decode the scaling and normalize the images
            pd_data = decode_preprocessing(pd_data, exptime, min_value, max_value)
            
            # Save as new fits file
            new_map = Map(pd_data, pd_meta)
            new_map.save(pd_fits_file, filetype='fits', overwrite=True)
