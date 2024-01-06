import argparse
import os
from glob import glob
import warnings

from evaluate_fulldisk import evaluate_fulldisk
from evaluate_activeregion import evaluate_activeregion

def main(args):
    gt_dirs = sorted(glob(os.path.join(args.source, 'gt_fits', '*')))
    pd_dirs = sorted(glob(os.path.join(args.source, 'pd_fits', '*')))

    if len(gt_dirs) == 0:
        raise ValueError('No ground truth directories found.')
    elif len(pd_dirs) == 0:
        raise ValueError('No prediction directories found.')
    elif len(gt_dirs) != len(pd_dirs):
        raise ValueError('The number of directories in gt_fits and pd_fits must be the same.')
    
    len_test   = len(pd_dirs)
    len_seq    = len(glob(os.path.join(gt_dirs[0], '*')))
    len_output = len(glob(os.path.join(pd_dirs[0], '*')))
    len_input  = len_seq - len_output
    
    if len_test == 0:
        raise ValueError('No test directories found.')
    elif len_seq == 0:
        raise ValueError('No files found in the ground truth directories.')
    elif len_output == 0:
        raise ValueError('No files found in the prediction directories.')

    if args.eval_type == 'fulldisk':
        evaluate_fulldisk(args.source, gt_dirs, pd_dirs, len_test, len_seq, len_output, len_input, args)
    elif args.eval_type == 'activeregion':
        evaluate_activeregion(args.source, gt_dirs, pd_dirs, len_test, len_seq, len_output, len_input, args)
    else:
        raise ValueError('Argument \'eval_type\' must be fulldisk or activeregion')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--eval_type', type=str, default='fulldisk')
    parser.add_argument('--dont_plot', action='store_true')
    parser.add_argument('--out_dir', default='.')
    args = parser.parse_args()

    warnings.simplefilter('ignore', RuntimeWarning)

    main(args)
