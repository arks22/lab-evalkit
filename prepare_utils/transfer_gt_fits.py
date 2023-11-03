# coding: utf-8
"""
割り当て表に従い、GTに相当するFITSファイルをコピーする
3波長の場合は -> 1波長のチャンネル吸い出しも行う
"""

import ast
import os
import shutil
from tqdm import tqdm

def txt_to_array_dataset(file_path, dataset_type):
    """
    指定されたタイプ（例: 'train', 'val' など）に対応する行を読み出し、Pythonのリスト形式に変換する。
    :param file_path: データセットの情報が含まれるテキストファイルのパス
    :param dataset_type: 読み出したいデータセットのタイプ（例: 'train', 'val'）
    :return: Pythonリスト形式に変換されたデータセット
    """

    # ファイルを開き、行ごとに読み出す
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 特定のデータセットタイプに対応する行を選択
    dataset_lines = [line for line in lines if line.startswith(dataset_type)]

    # 選択された行をPythonのリストに変換
    datasets = []
    for line in dataset_lines:
        # 'train'などのタイプの文字を取り除き、リスト形式の部分だけを取り出す
        list_str = line[len(dataset_type):].strip()
        # 文字列をPythonのリストに変換
        dataset = ast.literal_eval(list_str)
        datasets.append(dataset)

    return datasets


def transfer_gt_fits(source, timestamp, assignment_path, wave_n):
    print('1. transfer ground truth FITS ....')
    dataset_type = 'test'

    npy_list = txt_to_array_dataset(assignment_path, dataset_type)
    print('Number of total files:', len(npy_list))

    for i, npy_seq in tqdm(enumerate(npy_list)):

        if wave_n == 3: # 3波長の場合211のリストのみを抽出
            npy_seq = npy_seq[0]

        # 拡張子を置換
        fits_seq = [filename.replace('npy', 'fits') for filename in npy_seq]

        # ターゲットディレクトリを作成
        target_dir = os.path.join(timestamp, 'gt_fits', f"{i+1:02}")
        os.makedirs(target_dir, exist_ok=True)

        # FITSファイルをコピー
        for fits in fits_seq:
            shutil.copy(os.path.join(source, fits), os.path.join(target_dir, fits))