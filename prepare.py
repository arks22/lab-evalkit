import argparse
import os
from os.path import join
from prepare_utils.transfer_gt_fits import transfer_gt_fits
from prepare_utils.transfer_pd_npy import transfer_pd_npy
from prepare_utils.reverse_processing import reverse_processing 
from prepare_utils.split_npy import split_npy
from prepare_utils.convert_npy_to_fits import convert_npy_to_fits


def main(args):
    # Make output directory
    parent_dir = 'aia211' if args.wave_n == 1 else 'aia3wave'
    dst_dir = join(parent_dir, args.pd_timestamp)
    tmp_dir = join(dst_dir, 'tmp')
    gt_fits_dir = join(dst_dir, 'gt_fits')
    pd_fits_dir = join(dst_dir, 'pd_fits')
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(gt_fits_dir, exist_ok=True)
    os.makedirs(pd_fits_dir, exist_ok=True)

    # 1. Transfer the ground truth fits files
    transfer_gt_fits(args.fits_source, gt_fits_dir, args.assignment_path, args.wave_n)

    # 2. Transfer the prediction npy files
    transfer_pd_npy(args.pd_timestamp, tmp_dir, dst_dir, args.wave_n)

    # 3. Decode the scaling and normalize the images
    reverse_processing(tmp_dir, args.min_value, args.max_value)
   
    # 4. Split the images
    split_npy(tmp_dir)

    # 5. Convert npy to fits
    convert_npy_to_fits(tmp_dir, gt_fits_dir, pd_fits_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pd_timestamp', type=str, help='TimeStamp of The predictiopn source directory from which to copy the npy files')
    parser.add_argument('--fits_source', type=str, required=True)
    parser.add_argument('--assignment_path', type=str, required=True)
    parser.add_argument('--min_value', type=float, default=0, help='Min value for decoding normalization')
    parser.add_argument('--max_value', type=float, default=10000, help='Max value for decoding normalization')
    parser.add_argument('--wave_n', type=float, default=1, help='Number of wavelength channels')
    
    args = parser.parse_args()
    main(args)