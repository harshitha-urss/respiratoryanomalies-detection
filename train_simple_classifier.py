import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load train/test data
X_train = np.load('mfcc_train.npy')
X_test = np.load('mfcc_test.npy')
y_train = np.load('labels_train.npy')
y_test = np.load('labels_test.npy')

# Reshape MFCC data for scikit-learn (flatten each feature vector)
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# Train a simple Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_flat, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_flat)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
