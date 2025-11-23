import tkinter as tk
from tkinter import messagebox
from threading import Thread
import sounddevice as sd
import wavio
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# ------ Settings ------
fs = 22050                     # Sample rate
audio_data = None
recording = False
filename = '../data/recorded_audio.wav'    # Update path if needed
model = load_model('../data/cnn_mfcc_model.h5')
target_names = ['normal', 'cough', 'wheeze']  # Update if you change your training classes

# ------ Functions ------
def start_recording():
    global audio_data, recording
    if recording:
        messagebox.showinfo("Info", "Recording already in progress.")
        return
    recording = True
    messagebox.showinfo("Recording", "Recording started! Speak now, and click 'Stop Recording' when done.")
    audio_data = sd.rec(int(60 * fs), samplerate=fs, channels=1)  # Record up to 60 seconds


def stop_recording():
    global audio_data, recording
    if not recording:
        messagebox.showinfo("Info", "Not currently recording.")
        return
    sd.stop()
    recording = False
    wavio.write(filename, audio_data, fs, sampwidth=2)

    # Extract MFCC features
    n_mfcc = 40
    max_pad_len = 173
    y, sr = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    input_data = mfcc[np.newaxis, ..., np.newaxis]

    # Model prediction
    pred = model.predict(input_data)
    pred_class = np.argmax(pred, axis=1)[0]
    result_class = target_names[pred_class]
    infection_status = 'Uninfected' if result_class == 'normal' else 'Infected'

    messagebox.showinfo("Prediction", f"Prediction: {infection_status}")

# ------ GUI Setup ------
root = tk.Tk()
root.title("Audio Recorder & Infection Detector")
root.geometry("350x200")

start_btn = tk.Button(root, text="Start Recording", command=lambda: Thread(target=start_recording).start(), width=20, height=2)
start_btn.pack(pady=20)

stop_btn = tk.Button(root, text="Stop Recording", command=lambda: Thread(target=stop_recording).start(), width=20, height=2)
stop_btn.pack(pady=10)

root.mainloop()
