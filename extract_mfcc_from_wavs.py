import librosa
import numpy as np
import pandas as pd
import os

# Path setup
metadata_file = r'G:\My Drive\Respiratory Anomalies\data\coughvid\coughvid_20211012\metadata_compiled.csv'
audio_folder = r'G:\My Drive\Respiratory Anomalies\data\coughvid\coughvid_20211012'

meta = pd.read_csv(metadata_file)

pairs = []
for idx, row in meta.iterrows():
    status = str(row['status']).lower()
    if status not in ['healthy', 'covid-19', 'symptomatic']:
        continue
    wav_path = os.path.join(audio_folder, row['uuid'] + '.wav')
    if os.path.exists(wav_path):
        label = 0 if status == 'healthy' else 1
        pairs.append((wav_path, label))
    if len(pairs) >= 50:  # First 50 files for speed
        break

n_mfcc = 40         # Number of MFCCs (change as needed)
max_pad_len = 173   # Fixed length for padding/truncation

mfcc_data = []
labels = []

# We assume 'pairs' is your list of (wav_path, label) from the previous step
for wav_path, label in pairs:
    try:
        y, sr = librosa.load(wav_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Pad or truncate to fixed length
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        mfcc_data.append(mfcc)
        labels.append(label)
    except Exception as e:
        print(f"Error with {wav_path}: {e}")

mfcc_data = np.array(mfcc_data)[..., np.newaxis]  # shape to [samples, n_mfcc, max_pad_len, 1]
labels = np.array(labels)  # shape to [samples]

print("MFCC shape:", mfcc_data.shape)
print("Labels shape:", labels.shape)
