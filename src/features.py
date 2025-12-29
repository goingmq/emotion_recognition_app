import os
import numpy as np
import librosa
from .config import SR, N_MFCC, MAX_PAD_LEN, EMOTIONS


def extract_emotion_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('.')[0].split('-')
    if len(parts) < 3:
        return None
    return EMOTIONS.get(parts[2])


def pad_or_truncate(mfcc, max_len=MAX_PAD_LEN):
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    stacked = np.vstack([mfcc, delta, delta2])
    stacked = pad_or_truncate(stacked)

    return stacked
