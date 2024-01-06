import os
import glob
import shutil

dir = '/mnt/hdd1/sasaki/MAU_data/1_2_fits/171'

files = sorted(glob.glob(f'{dir}/*2022-10-28T23*.fits'))
files.append(sorted(glob.glob(f'{dir}/*2022-10-29*.fits')))
files.append(sorted(glob.glob(f'{dir}/*2022-10-30*.fits')))
files.append(sorted(glob.glob(f'{dir}/*2022-10-31*.fits')))
files.append(sorted(glob.glob(f'{dir}/*2022-11-01*.fits')))

files = sum(files, [])

for file in files:
    shutil.copy(file, './concept_exp2/171/')
    
##########################################

dir = '/mnt/hdd1/sasaki/MAU_data/1_2_fits/193'

files = sorted(glob.glob(f'{dir}/*2022-10-28T23*.fits'))
files.append(sorted(glob.glob(f'{dir}/*2022-10-29*.fits')))
files.append(sorted(glob.glob(f'{dir}/*2022-10-30*.fits')))
files.append(sorted(glob.glob(f'{dir}/*2022-10-31*.fits')))
files.append(sorted(glob.glob(f'{dir}/*2022-11-01*.fits')))

files = sum(files, [])

for file in files:
    shutil.copy(file, './concept_exp2/193/')

##########################################

dir = '/mnt/hdd1/sasaki/MAU_data/1_2_fits/211'

files = [sorted(glob.glob(f'{dir}/*2022-10-28T23*.fits'))]
files.append(sorted(glob.glob(f'{dir}/*2022-10-29*.fits')))
files.append(sorted(glob.glob(f'{dir}/*2022-10-30*.fits')))
files.append(sorted(glob.glob(f'{dir}/*2022-10-31*.fits')))
files.append(sorted(glob.glob(f'{dir}/*2022-11-01*.fits')))

files = sum(files, [])

for file in files:
    shutil.copy(file, './concept_exp2/211/')