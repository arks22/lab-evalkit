# coding: utf-8
"""
割り当て表に従い、GTに相当するFITSファイルをコピーする
"""

import re
import ast
import os
import shutil
import argparse
from datetime import datetime
from tqdm import tqdm

def transfer_gt_fits(source, timestamp, assignment_path):
    print('1. transfer ground truth FITS ....')
    dataset_type = 'test'

    # new_dataset_date_assignment.txtを読み込む
    with open(assignment_path, 'r') as f:
        lines = f.readlines()

    # 引数で指定された種類の行を抽出
    dataset_lines = [line for line in lines if line.startswith(dataset_type)]
    fits_list = []

    # 各行について処理
    for i, line in tqdm(enumerate(dataset_lines)):
        # スペースで区切られた要素を取得
        elements = line.split()

        text = re.search(r'\[(.*?)\]', line).group(1)
        array = ast.literal_eval(text)
        array = [filename.replace('npy', 'fits') for filename in array]

        # ターゲットディレクトリを作成
        target_dir = os.path.join(timestamp, 'gt_fits', f"{i+1:02}")
        os.makedirs(target_dir, exist_ok=True)

        # FITSファイルをコピー
        for fits in array:
            shutil.copy(os.path.join(source, fits), os.path.join(target_dir, fits))
