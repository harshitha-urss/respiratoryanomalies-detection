import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.ensemble import RandomForestClassifier

# ---- Config ----
AUDIO_FILE = 'recorded_audio.wav'  # path to your live or test .wav file
N_MFCC = 40
MAX_PAD_LEN = 173
DURATION = 5  # seconds to record from mic
FS = 22050   # sampling rate expected by librosa.load

# ---- Record audio from mic ----
print("Recording from microphone for {} seconds...".format(DURATION))
audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
sd.wait()
audio = audio.flatten()
write(AUDIO_FILE, FS, audio)
print(f"Audio saved to {AUDIO_FILE}")

# ---- Load pre-trained model ----
# (load the same model type used for training)
X_train = np.load('mfcc_train.npy')
y_train = np.load('labels_train.npy')
X_train_flat = X_train.reshape(len(X_train), -1)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_flat, y_train)

# ---- Extract MFCC from input audio ----
y, sr = librosa.load(AUDIO_FILE, sr=None)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
pad_width = MAX_PAD_LEN - mfcc.shape[1]
if pad_width > 0:
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
else:
    mfcc = mfcc[:, :MAX_PAD_LEN]

# ---- Predict ----
# ---- Predict ----
mfcc_flat = mfcc.reshape(1, -1)

proba = clf.predict_proba(mfcc_flat)[0]
print("Class probabilities [HEALTHY, INFECTED]:", proba)

prediction = np.argmax(proba)
label = 'HEALTHY' if prediction == 0 else 'INFECTED'
print(f"Prediction for the audio: {label}")
