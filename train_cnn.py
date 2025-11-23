import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Load MFCC features and labels
X_train = np.load('../data/features/X_train_mfcc.npy')
y_train = np.load('../data/features/y_train_mfcc.npy')
X_test = np.load('../data/features/X_test_mfcc.npy')
y_test = np.load('../data/features/y_test_mfcc.npy')

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)

# 2. Reshape for CNN (add channel axis)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# 3. One-hot encode the labels
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# 4. Build a simple CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 5. Train the model
history = model.fit(
    X_train, y_train_cat,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test_cat)
)

# 6. Evaluate on test data
loss, acc = model.evaluate(X_test, y_test_cat)
print(f'\nTest accuracy: {acc:.3f}')

model.save('../data/cnn_mfcc_model.h5')
print('Model saved!')