import numpy as np
import os
import glob
from tqdm import tqdm

def split_npy(tmp_dir):
    print('4. split npy...')

    # Get all npy files in the tmp directory
    for npy_file in tqdm(glob.glob(f'{tmp_dir}/*.npy')):
        # Load the numpy array
        npy_data = np.load(npy_file)

        # Make a new directory with the same name as the npy file
        dir_path = os.path.join(tmp_dir, os.path.splitext(os.path.basename(npy_file))[0])
        os.makedirs(dir_path, exist_ok=True)

        # Save each image in the numpy array as a new npy file
        for i in range(npy_data.shape[0]):
            new_npy_file = os.path.join(dir_path, f"{i+1:02d}.npy")
            np.save(new_npy_file, npy_data[i])

        # Remove the original npy file
        os.remove(npy_file)
