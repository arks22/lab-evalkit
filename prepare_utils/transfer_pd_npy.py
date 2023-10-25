import shutil
import glob
import os
from tqdm import tqdm

def transfer_pd_npy(timestamp):
    print('2. transfer prediction npy....')

    # Copy all npy files from the source directory to the tmp directory
    path = '/home/sasaki/MAU/results/aia211/' + timestamp + '/test/ndarray/*'

    for npy_file in tqdm(glob.glob(path)):
        shutil.copy(npy_file, f'{timestamp}/tmp/')
