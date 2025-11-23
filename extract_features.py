import librosa
import numpy as np
import pandas as pd
import os

base_dirs = {'train': '../data/train', 'test': '../data/test'}
output_dir = '../data/features'
os.makedirs(output_dir, exist_ok=True)

# Feature settings
n_mfcc = 40
max_pad_len = 173  # pad/truncate to uniform length for CNN

for split, data_dir in base_dirs.items():
    print(f'Extracting MFCCs from {split} set...')
    df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    features = []
    for idx, row in df.iterrows():
        file_path = os.path.join(data_dir, row['filename'])
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        features.append(mfcc)
    X = np.stack(features)
    y = df['label'].factorize()[0]
    np.save(os.path.join(output_dir, f'X_{split}_mfcc.npy'), X)
    np.save(os.path.join(output_dir, f'y_{split}_mfcc.npy'), y)
print('Feature extraction complete!')
