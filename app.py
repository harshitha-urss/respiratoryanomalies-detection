from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Enable CORS for all origins (for testing, you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Netlify URL later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"msg": "API is running"}

# ---- Load and train the model at startup ----
N_MFCC = 40
MAX_PAD_LEN = 173

X_train = np.load('mfcc_train.npy')
y_train = np.load('labels_train.npy')
X_train_flat = X_train.reshape(len(X_train), -1)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_flat, y_train)

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    contents = await file.read()
    # Save the uploaded audio file temporarily
    with open("temp.wav", "wb") as f:
        f.write(contents)

    # Extract MFCC features
    y, sr = librosa.load("temp.wav", sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    pad_width = MAX_PAD_LEN - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]

    mfcc_flat = mfcc.reshape(1, -1)
    proba = clf.predict_proba(mfcc_flat)[0]
    prediction = np.argmax(proba)
    label = 'HEALTHY' if prediction == 0 else 'INFECTED'
    return JSONResponse(content={"label": label, "probabilities": proba.tolist()})
