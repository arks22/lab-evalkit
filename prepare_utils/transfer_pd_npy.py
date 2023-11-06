# MAUの予測結果のnpyファイルをコピーしてくる
# 3波長の場合は1波長のチャンネル吸い出しも行う
import shutil
import glob
import os
import numpy as np
from tqdm import tqdm


def transfer_pd_npy(timestamp, tmp_dir, dst_dir, wave_n):
    print('2. transfer prediction npy....')

    if wave_n == 1:
        wave_dir = '/home/sasaki/MAU/results/aia211'
    elif wave_n == 3:
        wave_dir = '/home/sasaki/MAU/results/aia3wave'
    else:
        raise ValueError('wave_n must be 1 or 3')

    """
    source_path = os.path.join(wave_dir, timestamp, 'test/ndarray/*')

    for npy_file in tqdm(glob.glob(path)):
        npy_path = shutil.copy(npy_file, tmp_dir)
    
        # 3波長の場合、コピーしたnpyファイルから1波長のチャンネルを抽出する
        if wave_n == 3:
            data = np.load(npy_path)
            data_1wave = data[:, :, :, 0:1] #チャンネルの次元を保持するためにスライスを使う
            np.save(npy_path, data_1wave)
    """
        
    # copy files
    print(dst_dir)
    copy_files = ['configs.json', 'results.json', 'losses.png']
    for file in copy_files:
        shutil.copy(os.path.join(wave_dir, timestamp, file), dst_dir)

timestamps = ['202310311754','202311012114']
tmp_dir = ""
wave_n = 3
for timestamp in timestamps:
    dst_dir = f'./aia3wave/{timestamp}'
    transfer_pd_npy(timestamp, tmp_dir, dst_dir, wave_n)
    print(f'Finished {timestamp}')

timestamps = ['202306161543','202311031847', '202311031850']
wave_n = 1
for timestamp in timestamps:
    dst_dir = f'./aia211/{timestamp}'
    transfer_pd_npy(timestamp, tmp_dir, dst_dir, wave_n)
    print(f'Finished {timestamp}')