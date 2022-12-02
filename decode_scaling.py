from glob import glob
import numpy as np
import argparse
import os
import cv2

parser = argparse.ArgumentParser(description='normalize fits')
parser.add_argument('source', type=str)
parser.add_argument('--out_dir', type=str, default='.')
parser.add_argument('--method', type=str)
parser.add_argument('--max', type=int, )
parser.add_argument('--min', type=int, )
parser.add_argument('--resize', type=int, required=True)
args = parser.parse_args()

npy_files  = sorted(glob('npy_predicted/*'))

def decode_sqrt(data):
    data = data * np.sqrt(args.max - args.min)
    data = data ** 2
    return data

for file in npy_files:
    video = np.load(file)
    video = decode_sqrt(video)
    resized_video = np.zeros((video.shape[0], args.resize, args.resize))

    for i in range(len(video)):
        frame = np.squeeze(video[i])
        frame = cv2.resize(frame, dsize=(args.resize, args.resize),interpolation=cv2.INTER_CUBIC)
        resized_video[i] = frame

    print(resized_video.shape)
    name = os.path.splitext(os.path.basename(file))[0]
    np.save(os.path.join(args.out_dir, name), resized_video)
