import numpy as np
import cv2
import glob
import os
from tqdm import tqdm


def reverse_processing(timestamp, min_value, max_value):
    print('3. reverse processing....')
    files = glob.glob(f'{timestamp}/tmp/*.npy')
    
    for file in tqdm(files):
        # Load the data
        data = np.load(file)
        
        # Decode the scaling
        data = np.square(data)

        # Decode the normalization
        data = data * (max_value - min_value) + min_value

        data = np.squeeze(data)
        
        # Save the file back to the 'tmp' directory
        np.save(os.path.join(timestamp, 'tmp', os.path.basename(file)), data)