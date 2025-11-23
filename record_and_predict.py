import sounddevice as sd
import wavio
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# === Step 1: Record audio ===
fs = 22050       # Sample rate
seconds = 4      # Duration
filename = '../data/recorded_audio.wav'

print("Please speak now (4 seconds)...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
wavio.write(filename, recording, fs, sampwidth=2)
print(f'Audio recorded and saved to {filename}')

# === Step 2: Extract MFCC Features ===
n_mfcc = 40
max_pad_len = 173

y, sr = librosa.load(filename, sr=None)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
pad_width = max_pad_len - mfcc.shape[1]
if pad_width > 0:
    mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
else:
    mfcc = mfcc[:, :max_pad_len]
input_data = mfcc[np.newaxis, ..., np.newaxis]

# === Step 3: Load Model and Predict ===
model = load_model('../data/cnn_mfcc_model.h5')
target_names = ['normal', 'cough', 'wheeze']  # your class labels

pred = model.predict(input_data)
pred_class = np.argmax(pred, axis=1)[0]
result_class = target_names[pred_class]

# Map multi-class prediction to binary infection status
if result_class == 'normal':
    infection_status = 'Uninfected'
else:
    infection_status = 'Infected'

print(f"Prediction: {infection_status}")
