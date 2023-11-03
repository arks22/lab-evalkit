# MAUの予測結果のnpyファイルをコピーしてくる
# 3波長の場合は1波長のチャンネル吸い出しも行う
import shutil
import glob
import os
import numpy as np
from tqdm import tqdm


def transfer_pd_npy(timestamp, wave_n):
    print('2. transfer prediction npy....')

    if wave_n == 1:
        # Copy all npy files from the source directory to the tmp directory
        path = '/home/sasaki/MAU/results/aia211/' + timestamp + '/test/ndarray/*'
    elif wave_n == 3:
        path = '/home/sasaki/MAU/results/aia3wave/' + timestamp + '/test/ndarray/*'
    else:
        raise ValueError('wave_n must be 1 or 3')

    # ディレクトリを作成
    os.makedirs(f'{timestamp}/tmp', exist_ok=True)

    for npy_file in tqdm(glob.glob(path)):
        npy_path = shutil.copy(npy_file, f'{timestamp}/tmp/')
    
        # 3波長の場合、コピーしたnpyファイルから1波長のチャンネルを抽出する
        if wave_n == 3:
            data = np.load(npy_path)
            data_1wave = data[:, :, :, 0:1] #チャンネルの次元を保持するためにスライスを使う
            np.save(npy_path, data_1wave)