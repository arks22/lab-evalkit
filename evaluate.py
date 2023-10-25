import argparse
import os
from glob import glob
import warnings

from evaluate_fulldisk import evaluate_fulldisk
from evaluate_activeregion import evaluate_activeregion


def main(args):
    gt_dirs = sorted(glob(f'{args.timestamp}/gt_fits/*'))
    pd_dirs = sorted(glob(f'{args.timestamp}/pd_fits/*'))
    
    len_test   = len(pd_dirs)
    len_seq    = len(glob(f'{gt_dirs[0]}/*'))
    len_output = len(glob(f'{pd_dirs[0]}/*'))
    len_input  = len_seq - len_output


    if args.eval_type == 'fulldisk':
        evaluate_fulldisk(args.timestamp, gt_dirs, pd_dirs, len_test, len_seq, len_output, len_input, args)

    elif args.eval_type == 'activeregion':
        evaluate_activeregion(args.timestamp, gt_dirs, pd_dirs, len_test, len_seq, len_output, len_input, args)

    else:
        raise ValueError('Argument \'eval_type\' must be fulldisk or activeregion or limb.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=str)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--eval_type', type=str, default='fulldisk')
    parser.add_argument('--dont_plot', action='store_true')
    args = parser.parse_args()

    warnings.simplefilter('ignore', RuntimeWarning)

    main(args)
