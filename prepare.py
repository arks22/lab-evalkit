import argparse
import os
from prepare_utils.transfer_gt_fits import transfer_gt_fits
from prepare_utils.transfer_pd_npy import transfer_pd_npy
from prepare_utils.reverse_processing import reverse_processing 
from prepare_utils.split_npy import split_npy
from prepare_utils.convert_npy_to_fits import convert_npy_to_fits


def main(args):
    # Make output directory
    os.makedirs(f'{args.pd_timestamp}/tmp', exist_ok=True)
    os.makedirs(f'{args.pd_timestamp}/gt_fits', exist_ok=True)
    os.makedirs(f'{args.pd_timestamp}/pd_fits', exist_ok=True)

    # 1. Transfer the ground truth fits files
    #transfer_gt_fits(args.fits_source, args.pd_timestamp, args.assignment_path)
    
    # 2. Transfer the prediction npy files
    transfer_pd_npy(args.pd_timestamp)
    
    # 3. Decode the scaling and normalize the images
    reverse_processing(args.pd_timestamp, args.min_value, args.max_value)
   
    # 4. Split the images
    split_npy(args.pd_timestamp)
    
    # 5. Convert npy to fits
    convert_npy_to_fits(args.pd_timestamp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pd_timestamp', type=str, help='TimeStamp of The predictiopn source directory from which to copy the npy files')
    parser.add_argument('--fits_source', type=str, required=True)
    parser.add_argument('--assignment_path', type=str, required=True)
    parser.add_argument('--min_value', type=float, default=0, help='Min value for decoding normalization')
    parser.add_argument('--max_value', type=float, default=10000, help='Max value for decoding normalization')
    
    args = parser.parse_args()
    main(args)
