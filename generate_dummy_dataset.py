import numpy as np
from scipy.io.wavfile import write
import os
import csv

# Dataset paths
base_path = '../data'
train_path = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')

# Audio parameters
sample_rate = 22050  # standard sample rate
sample_duration = 2  # seconds

# Classes and their base frequencies (Hz)
classes = {
    'normal': 440,   # tone A4
    'cough': 1000,   # higher pitch
    'wheeze': 600    # medium pitch with tremolo
}

# Number of files per class per split
num_train_per_class = 533
num_test_per_class = 133

def generate_tone(freq, duration, sample_rate, label):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    if label == 'cough':
        tone *= (np.random.rand(*tone.shape) > 0.7)  # bursts
    elif label == 'wheeze':
        tone *= (1.0 + 0.2 * np.sin(2 * np.pi * 8 * t))  # tremolo
    return tone.astype(np.float32)


def write_dataset(split_path, num_per_class):
    os.makedirs(split_path, exist_ok=True)
    labels = []
    for label, freq in classes.items():
        for i in range(num_per_class):
            tone = generate_tone(freq, sample_duration, sample_rate, label)
            filename = f'{label}_{i+1}.wav'
            path = os.path.join(split_path, filename)
            write(path, sample_rate, tone)
            labels.append([filename, label])
    # Write labels CSV
    with open(os.path.join(split_path, 'labels.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(labels)


if __name__ == '__main__':
    print('Generating training dataset...')
    write_dataset(train_path, num_train_per_class)
    print('Generating testing dataset...')
    write_dataset(test_path, num_test_per_class)
    print('Dummy audio dataset generation complete.')
