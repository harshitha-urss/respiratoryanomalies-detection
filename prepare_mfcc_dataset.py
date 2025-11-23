import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

# ---- 1. Setup paths ----
metadata_file = r'G:\My Drive\Respiratory Anomalies\data\coughvid\coughvid_20211012\metadata_compiled.csv'
audio_folder = r'G:\My Drive\Respiratory Anomalies\data\coughvid\coughvid_20211012'

# ---- 2. Load metadata and pair files ----
meta = pd.read_csv(metadata_file)

pairs = []
for idx, row in meta.iterrows():
    status = str(row['status']).lower()
    if status not in ['healthy', 'covid-19', 'symptomatic']:
        continue
    wav_path = os.path.join(audio_folder, row['uuid'] + '.wav')
    if os.path.exists(wav_path):
        label = 0 if status == 'healthy' else 1        # Uninfected = 0, Infected = 1
        pairs.append((wav_path, label))
    if len(pairs) >= 1000:                               # Limit to first 50 for speed
        break

print(f"Using {len(pairs)} audio files for MFCC extraction.")

# ---- 3. Extract MFCC features ----
n_mfcc = 40
max_pad_len = 173

mfcc_data = []
labels = []

for i, (wav_path, label) in enumerate(pairs, 1):
    try:
        y, sr = librosa.load(wav_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        mfcc_data.append(mfcc)
        labels.append(label)
        print(f"Processed {i}/{len(pairs)}: {wav_path}")
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")

mfcc_data = np.array(mfcc_data)[..., np.newaxis]   # shape: (samples, n_mfcc, max_pad_len, 1)
labels = np.array(labels)

unique, counts = np.unique(labels, return_counts=True)
print("Class counts:", dict(zip(unique, counts)))

# Balance the dataset: downsample the majority (healthy) class
healthy_idx = np.where(labels == 0)[0]
infected_idx = np.where(labels == 1)[0]
minority_count = min(len(healthy_idx), len(infected_idx))

# Randomly select same number of samples from both classes
np.random.seed(42)
healthy_sample = np.random.choice(healthy_idx, minority_count, replace=False)
infected_sample = np.random.choice(infected_idx, minority_count, replace=False)

# Combine, shuffle, and create new balanced arrays
final_indices = np.concatenate([healthy_sample, infected_sample])
np.random.shuffle(final_indices)
mfcc_data_balanced = mfcc_data[final_indices]
labels_balanced = labels[final_indices]

# Check new class counts
unique, counts = np.unique(labels_balanced, return_counts=True)
print("Balanced class counts:", dict(zip(unique, counts)))

print("Final MFCC shape:", mfcc_data.shape)
print("Labels shape:", labels.shape)

# ---- 4. Train/test split ----
# ---- 4. Train/test split on balanced data (and stratify for fairness) ----
X_train, X_test, y_train, y_test = train_test_split(
    mfcc_data_balanced, labels_balanced, test_size=0.2, random_state=42, stratify=labels_balanced
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# ---- 5. Save data for later model training ----
np.save('mfcc_train.npy', X_train)
np.save('mfcc_test.npy', X_test)
np.save('labels_train.npy', y_train)
np.save('labels_test.npy', y_test)
print("Saved train/test splits as .npy files.")
