import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('../data/cnn_mfcc_model.h5')

# Parameters (must match training)
n_mfcc = 40
max_pad_len = 173

# Map predicted class indices to labels
target_names = ['normal', 'cough', 'wheeze']

# Replace with the path to a test audio file
file_path = '../data/test/normal_1.wav'

# Load and preprocess

y, sr = librosa.load(file_path, sr=None)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
pad_width = max_pad_len - mfcc.shape[1]
if pad_width > 0:
    mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
else:
    mfcc = mfcc[:, :max_pad_len]

# Reshape for CNN
input_data = mfcc[np.newaxis, ..., np.newaxis]

# Predict
pred = model.predict(input_data)
pred_class = np.argmax(pred, axis=1)[0]
print(f'Predicted class: {target_names[pred_class]}')
